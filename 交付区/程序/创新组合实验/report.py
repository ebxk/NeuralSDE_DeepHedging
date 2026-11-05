"""Replay all frozen policies and publish aggregate and path-level evidence."""
import csv
from collections import defaultdict
import gzip
from itertools import combinations
import json
from pathlib import Path
import sys
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from bs_baseline import delta
from contract_ledger import EVENTS,ledger
from hedge_experiment import read_paths
from prepare_common import digest,write_json
from train import (expected_shortfall,group_path,make_model,make_real,replay,tasks)


def interpolation(tau,k,ts,ks,values):
    """Bilinear value and strike slope, with a maturity-zero intrinsic row."""
    t=np.clip(tau,ts[0],ts[-1])
    q=np.clip(k,ks[0],ks[-1])
    i=np.clip(np.searchsorted(ts,t)-1,0,len(ts)-2)
    j=np.clip(np.searchsorted(ks,q)-1,0,len(ks)-2)
    x=(t-ts[i])/(ts[i+1]-ts[i])
    y=(q-ks[j])/(ks[j+1]-ks[j])
    left=(1-x)*values[i,j]+x*values[i+1,j]
    right=(1-x)*values[i,j+1]+x*values[i+1,j+1]
    slope=(right-left)/(ks[j+1]-ks[j]) if ks[0]<k<ks[-1] else 0.
    return left*(1-y)+right*y,slope


def local_mv(path,common,rep,date_idx):
    ts=np.r_[0,np.unique(rep["grid_t"])]
    ks=np.unique(rep["grid_k"])
    xi=common["real_xi"]
    cov=common["real_cov"]
    basis=rep["basis"]*common["scale"][:,None]
    basis=np.concatenate([np.zeros((2,1,len(ks))),basis.reshape(2,len(ts)-1,len(ks))],axis=1)
    hedge=[]
    for row in path[:-1]:
        i=date_idx[row["date"]]
        s,k=float(row["spot"]),float(row["strike"])
        tau=(np.datetime64(row["expiry"])-np.datetime64(row["date"])).astype(float)/365
        values=np.r_[np.maximum(1-ks,0),rep["center"]+(xi[i]*common["scale"])@rep["basis"]].reshape(len(ts),len(ks))
        c,dc=interpolation(tau,k/s,ts,ks,values)
        derivative=c-k/s*dc
        factor=np.array([interpolation(tau,k/s,ts,ks,b)[0] for b in basis])
        # At non-knot points the analytic spot derivative agrees with a central difference.
        eps=1e-6*s
        if np.min(np.abs(k/s-ks))>2e-6:
            plus=(s+eps)*interpolation(tau,k/(s+eps),ts,ks,values)[0]
            minus=(s-eps)*interpolation(tau,k/(s-eps),ts,ks,values)[0]
            assert abs(derivative-(plus-minus)/(2*eps))<1e-7
        assert cov[i,0,0]>1e-12
        hedge.append(float(np.clip(derivative+factor@cov[i,1:,0]/cov[i,0,0],0,1)))
    return hedge


def export(out,key,paths,books,cost,meta):
    metrics=[]
    dest=out/"daily"/f"{key}.csv.gz"
    tmp=dest.with_suffix(".tmp")
    fields=["episode","spot","strike","unit","factor1","factor2",*books[0][0]]
    date_to_xi=meta.pop("date_to_xi",{})
    with gzip.open(tmp,"wt",encoding="utf8",newline="",compresslevel=1) as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        for path,book in zip(paths,books):
            ep=path[0]
            assert len(path)==len(book) and book[-1]["date"]==ep["expiry"]
            nominal=float(ep["spot"])*int(ep["unit"])
            gates=[r.get("gate_weight","") for r in book[:-1]]
            gates=[g for g in gates if g!=""]
            for row,trade in zip(path,book):
                xi=date_to_xi.get(row["date"],("",""))
                w.writerow(dict(episode=ep["episode"],spot=row["spot"],strike=row["strike"],
                                unit=row["unit"],factor1=xi[0],factor2=xi[1],**trade))
            metrics.append(dict(episode=ep["episode"],contract=ep["code"],expiry=ep["expiry"],
                term=ep["term"],moneyness=ep["moneyness"],
                cross_div=any(ep["date"]<effective<=ep["expiry"] for effective,*_ in EVENTS),
                adjusted=any(int(r["unit"])!=10000 for r in path),
                pnl=float(book[-1]["wealth"]/nominal),cost=float(sum(r["cost"] for r in book)/nominal),
                turnover=float(sum(abs(r["traded"])*float(s["spot"]) for r,s in zip(book,path))/nominal),
                gate_mean=float(np.mean(gates)) if gates else "",
                gate_sd=float(np.std(gates)) if gates else ""))
    tmp.replace(dest)
    dest=out/"paths"/f"{key}.json"
    write_json(dest,dict(meta=meta,rows=metrics,daily_sha256=digest(out/"daily"/f"{key}.csv.gz")))
    return dest


def baseline_book(paths,ratios,cost):
    out=[]
    for p,r in zip(paths,ratios):
        b=ledger(p,r,float(p[0]["option_close"]),cost/10000)
        for row in b:
            row["gate_weight"]=""
        out.append(b)
    return out


def bs_path(path):
    iv=.25
    hedge=[]
    for row in path[:-1]:
        value,iv,_=delta(row,iv)
        hedge.append(value)
    return hedge


def frozen_replay(root,cfg,out):
    registry=json.loads((root/"registry.json").read_text(encoding="utf8"))
    groups=tasks(cfg)
    assert groups==registry["groups"]
    frozen=[]
    failures=[]
    for g in groups:
        p=group_path(root,g)
        for name in g["models"]:
            f=p/f"{name}.json"
            r=json.loads(f.read_text(encoding="utf8")) if f.exists() else dict(status="missing")
            if r["status"]=="complete":
                w=p/f"{name}.pt"
                assert digest(w)==r["weights_sha256"]
                identity=r["identity"]
                fold=root/str(g["end"])
                assert identity["token"]==registry["source"]
                assert identity["parent"]==digest(fold/g["rep"]/"market.done.json")
                assert identity["episodes"]==digest(fold/"episodes.done.json")
                assert all(digest(p/f"{expert}.pt")==sha for expert,sha in identity["experts"].items())
                frozen.append(dict(group=g,model=name,epoch=r["epoch"],sha256=r["weights_sha256"]))
            else:
                market=root/str(g["end"])/g["rep"]
                predeclared=(g["end"]!=2025 and g["market"] in {"equal","robust"}
                    and (market/"market.done.json").exists()
                    and json.loads((market/"market.json").read_text(encoding="utf8"))["members"]<2
                    and not (market/"pool.npz").exists())
                reason="Fewer than two calibrated market members" if predeclared else r.get("reason",r["status"])
                failures.append(dict(group=g,model=name,reason=reason,
                    status="unavailable" if predeclared else "failed"))
    write_json(out/"frozen_checkpoints.json",frozen)
    write_json(out/"failures.json",failures)
    if not frozen:
        raise RuntimeError("No policy checkpoints available")
    prepared={}
    records=[]
    for g in groups:
        end,rep=g["end"],g["rep"]
        p=group_path(root,g)
        folder=root/str(end)
        if not (folder/rep/"market.done.json").exists():
            continue
        if (end,rep) not in prepared:
            market=folder/rep
            manifest=json.loads((market/"market.done.json").read_text(encoding="utf8"))
            assert all(digest(market/name)==sha for name,sha in manifest["outputs"].items())
            source=json.loads((folder/"episodes.done.json").read_text(encoding="utf8"))
            assert digest(folder/"episode_days.csv")==source["outputs"]["episode_days.csv"]
            common=dict(np.load(market/"common.npz"))
            paths=read_paths(folder/"episode_days.csv",[end])[0]
            if end==2025:
                assert len(paths)==773
            inputs=make_real(paths,common,folder/"episode_days.csv")
            prepared[end,rep]=paths,inputs,common
        paths,inputs,common=prepared[end,rep]
        xi=dict(zip(common["real_dates"],common["real_xi"]))
        experts={}
        expert_seconds={}
        for name in g["models"]+(["fixed50","static"] if "factor" in g["models"] else []):
            started=time.perf_counter()
            fixed=None
            if name in {"fixed50","static"}:
                if set(experts)!={"base","factor"} or not (p/"static.json").exists():
                    continue
                fixed=.5 if name=="fixed50" else json.loads((p/"static.json").read_text(encoding="utf8"))["weight"]
                r=dict(seconds=0,parameters=sum(x.numel() for m in experts.values() for x in m.parameters()))
            else:
                stat=p/f"{name}.json"
                if not stat.exists():
                    continue
                r=json.loads(stat.read_text(encoding="utf8"))
                if r["status"]!="complete":
                    continue
            model=make_model(name,experts,fixed=fixed)
            if fixed is None:
                model.load_state_dict(torch.load(p/f"{name}.pt",weights_only=True))
            if name in {"base","factor"}:
                experts[name]=model
                expert_seconds[name]=r["seconds"]
            key=f"{end}_{p.name}_{name}"
            manifest=out/"paths"/f"{key}.json"
            meta={k:v for k,v in g.items() if k!="models"}
            total_seconds=r["seconds"]+(sum(expert_seconds.values()) if name.endswith("_gate") or fixed is not None else 0)
            meta.update(policy=name,parameters=r["parameters"],seconds=r["seconds"],standalone_seconds=total_seconds,
                        evaluation_identity=dict(weight=r.get("weights_sha256"),
                         experts=[digest(p/f"{x}.pt") for x in experts] if fixed is not None else [],
                         fixed=fixed,source=digest(Path(__file__)),policy_source=digest(Path(__file__).with_name("policy.py")),
                         market=digest(folder/rep/"market.done.json"),episodes=digest(folder/"episodes.done.json")))
            if manifest.exists() and (out/"daily"/f"{key}.csv.gz").exists():
                old=json.loads(manifest.read_text(encoding="utf8"))
                if (old["meta"].get("evaluation_identity")==meta["evaluation_identity"]
                    and old.get("daily_sha256")==digest(out/"daily"/f"{key}.csv.gz")):
                    records.append(manifest)
                    continue
            _,books=replay(model,paths,inputs,g["cost"]/10000,capture=True)
            meta.update(replay_seconds=time.perf_counter()-started,date_to_xi=xi)
            records.append(export(out,key,paths,books,g["cost"],meta))
    for (end,rep),(paths,_,common) in prepared.items():
        costs=[0,5,10] if end==2025 and rep in {"C0","C2"} else [5]
        xi=dict(zip(common["real_dates"],common["real_xi"]))
        data=dict(np.load(root/str(end)/f"{rep}.npz"))
        date_idx=dict(zip(common["real_dates"],range(len(common["real_dates"]))))
        methods={}
        if rep=="C0":
            methods["BS Delta"]=[bs_path(p) for p in paths]
        try:
            methods["Local MV"]=[local_mv(p,common,data,date_idx) for p in paths]
        except (AssertionError,ValueError,FloatingPointError) as exc:
            failures.append(dict(fold=end,rep=rep,model="Local MV",reason=repr(exc),status="baseline_unavailable"))
        for name,ratios in methods.items():
            for cost in costs:
                key=f"{end}_{rep}_{cost}bp_{name.replace(' ','_')}"
                meta=dict(end=end,rep=rep,cost=cost,market="baseline",lam=-1,seed=-1,
                          policy=name,parameters=0,seconds=0,standalone_seconds=0,replay_seconds=0,date_to_xi=xi)
                records.append(export(out,key,paths,baseline_book(paths,ratios,cost),cost,meta))
    write_json(out/"failures.json",failures)
    return records


def metrics(pnl):
    return np.sqrt(np.mean(pnl*pnl,axis=-1)),expected_shortfall(-pnl)


def write_table(path,rows):
    if not rows:
        return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",encoding="utf8",newline="") as file:
        w=csv.DictWriter(file,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def study_name(rep,market,policy):
    if rep not in {"C0","C2"} or market not in {"single","robust"} or policy not in {"factor","reliable_gate"}:
        return ""
    label=("A" if policy=="reliable_gate" else "")+("B" if market=="robust" else "")+("C" if rep=="C2" else "")
    return "+".join(label) if label else "Reference"


def blocks(months,length,reps,seed):
    bins=[np.flatnonzero(months==m) for m in np.unique(months)]
    n=len(bins)
    rng=np.random.default_rng(seed)
    for _ in range(reps):
        order=(rng.integers(n-min(n,length)+1,size=int(np.ceil(n/length)))[:,None]
               +np.arange(min(n,length))).ravel()[:n]
        yield np.concatenate([bins[i] for i in order])


def paired(pairs,labels,months,boot):
    if not pairs:
        return []
    a,b=np.stack([x for x,_ in pairs]),np.stack([y for _,y in pairs])
    rows=[]
    for length in (1,3,6):
        draws=np.empty((boot,len(labels),2))
        for i,idx in enumerate(blocks(months,length,boot,1105)):
            v1,v2=metrics(a[:,:,idx]),metrics(b[:,:,idx])
            for j in (0,1):
                draws[i,:,j]=(v1[j]-v2[j]).mean(axis=1)
        quant=np.quantile(draws,[.025,.975],axis=0)
        v1,v2=metrics(a),metrics(b)
        for i,label in enumerate(labels):
            for j,name in enumerate(("rmse","cvar95")):
                rows.append(dict(**label,block_months=length,metric=name,
                    delta=float(np.mean(v1[j][i]-v2[j][i])),
                    low=float(quant[0,i,j]),high=float(quant[1,i,j])))
    return rows


def summarize(out,records,cfg):
    seed_rows, path_catalog, summaries, comparisons = [], [], [], []
    strategies=defaultdict(dict)
    for file in records:
        result=json.loads(file.read_text(encoding="utf8"))
        m,p=result["meta"],result["rows"]
        rep,market,policy=m["rep"],m["market"],m["policy"]
        combo=study_name(rep,market,policy)
        key=(m["end"],m["cost"],m["lam"],rep,market,policy)
        groups={"all":np.ones(len(p),dtype=bool)}
        for name in ("term","moneyness","cross_div","adjusted","expiry"):
            values=np.array([r[name][:7] if name=="expiry" else r[name] for r in p])
            for value in np.unique(values):
                groups[f"{name}={value}"]=values==value
        pnl=np.array([r["pnl"] for r in p])
        strategies[key][m["seed"]]=(pnl,[r["episode"] for r in p],np.array([r["expiry"][:7] for r in p]))
        for label,mask in groups.items():
            x=pnl[mask]
            rmse,cvar=metrics(x)
            subset=[r for r,keep in zip(p,mask) if keep]
            gate=[r["gate_mean"] for r in subset if r["gate_mean"]!=""]
            seed_rows.append(dict(end=m["end"],cost=m["cost"],lam=m["lam"],rep=rep,market=market,
                policy=policy,combo=combo,seed=m["seed"],group=label,n=len(x),
                rmse=rmse,cvar95=cvar,mean_pnl=x.mean(),std=x.std(),
                trade_cost=np.mean([r["cost"] for r in subset]),turnover=np.mean([r["turnover"] for r in subset]),
                gate_mean=np.mean(gate) if gate else "",gate_sd=np.mean([r["gate_sd"] for r in subset if r["gate_sd"]!=""]) if gate else "",
                parameters=m["parameters"],seconds=m["seconds"],standalone_seconds=m.get("standalone_seconds",m["seconds"]),replay_seconds=m["replay_seconds"]))
        path_catalog.append(dict(id=file.stem,**{k:m[k] for k in ("end","cost","lam","rep","market","policy","seed")}))
    write_table(out/"metrics_by_seed.csv",seed_rows)
    group_rows=defaultdict(list)
    fixed=("end","cost","lam","rep","market","policy","combo","group")
    for row in seed_rows:
        group_rows[tuple(row[k] for k in fixed)].append(row)
    for keys,rs in group_rows.items():
        nums=("rmse","cvar95","mean_pnl","std","trade_cost","turnover","parameters","seconds","standalone_seconds","replay_seconds")
        row=dict(zip(fixed,keys),seeds=len(rs),n=rs[0]["n"])
        for k in nums:
            vals=[r[k] for r in rs]
            row.update({k:np.mean(vals),k+"_min":min(vals),k+"_max":max(vals)})
        summaries.append(row)
    write_table(out/"summary.csv",summaries)
    write_json(out/"paths_index.json",path_catalog)
    # All within-fold core comparisons, plus direct mechanism comparisons.
    for end,cost,lam in sorted(set(k[:3] for k in strategies if k[2]>=0)):
        labels,pairs=[],[]
        root=(end,cost,lam)
        cond=[k for k in strategies if k[:3]==root]
        core=[k for k in cond if study_name(*k[3:])]
        planned=[]
        for a,b in combinations(core,2):
            aset=set(study_name(*a[3:]).replace("+","")) if study_name(*a[3:])!="Reference" else set()
            bset=set(study_name(*b[3:]).replace("+","")) if study_name(*b[3:])!="Reference" else set()
            planned.append((b,a) if aset<bset else (a,b))
        for a in cond:
            _,_,_,rep,market,policy=a
            if policy=="reliable_gate":
                planned.extend((a,(*root,rep,market,x)) for x in ("static","state_gate","same_inputs","same_capacity","residual_gate","distance_gate","constant_gate"))
            if market=="robust":
                planned.append((a,(*root,rep,"equal",policy)))
            if rep=="C2":
                planned.append((a,(*root,"C1",market,policy)))
        for a in core:
            planned.extend((a,b) for b in [(*root,"C0","gbm","base"),
                (end,cost,-1,"C0","baseline","BS Delta"),(end,cost,-1,a[3],"baseline","Local MV")])
        for a,b in dict.fromkeys(planned):
            if a not in strategies or b not in strategies:
                continue
            aa,bb=strategies[a],strategies[b]
            if set(aa)!={0,1,2} or set(bb) not in ({0,1,2},{-1}):
                continue
            arr=[]
            for seed in (0,1,2):
                x,ids,months=aa[seed]
                y,yids,ym=bb.get(seed,bb.get(-1))
                assert ids==yids and np.array_equal(months,ym)
                arr.append((x,y))
            pairs.append(tuple(np.stack(v) for v in zip(*arr)))
            labels.append(dict(end=end,cost=cost,lam=lam,a="/".join(a[3:]),b="/".join(b[3:])))
        comparisons.extend(paired(pairs,labels,months if pairs else None,cfg["bootstrap"]))
    write_table(out/"paired_intervals.csv",comparisons)
    return summaries,comparisons


def figures(out,summary):
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,lam in zip(axes,[0,.1]):
        rows=[r for r in summary if (r["end"],r["cost"],r["lam"],r["group"])==(2025,5,lam,"all") and r["combo"]]
        for r in rows:
            ax.scatter(100*r["rmse"],100*r["cvar95"])
            ax.annotate(r["combo"],(100*r["rmse"],100*r["cvar95"]),xytext=(5,4),textcoords="offset points")
        ax.set(xlabel="RMSE (%)",ylabel="CVaR95 loss (%)",title=f"Historical evaluation, lambda={lam}")
    fig.tight_layout()
    fig.savefig(out/"risk_tradeoff.png",dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for ax,end in zip(axes.flat,[2021,2022,2023,2025]):
        rows=[r for r in summary if r["end"]==end and r["cost"]==5 and r["lam"]==0 and r["group"]=="all" and r["combo"]]
        ax.bar([r["combo"] for r in rows],[100*r["rmse"] for r in rows])
        ax.set(title=f"Evaluation ending {end}",ylabel="RMSE (%)")
    fig.tight_layout()
    fig.savefig(out/"rolling_stability.png",dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for ax,name in zip(axes,("rmse","cvar95")):
        rows=[r for r in summary if (r["end"],r["cost"],r["lam"],r["group"])==(2025,5,0,"all") and r["combo"]]
        ax.bar([r["combo"] for r in rows],[100*r[name] for r in rows])
        ax.set(ylabel=f"{name.upper()} (%)",title="2024–2025, 5bp, MSE")
    fig.tight_layout()
    fig.savefig(out/"comparison_5bp.png",dpi=180)
    plt.close(fig)


def diagnostics(root,out,cfg):
    rows=[]
    for fold in cfg["folds"]:
        f=root/str(fold[-1])
        for rep in ("C0","C1","C2"):
            if not (f/rep/"market.done.json").exists():
                continue
            with np.load(f/f"{rep}.npz") as d:
                dates=d["dates"]
                train=dates<str(fold[0]+1)
                a=d["xi"]@d["W"].T-d["bounds"]
                assert a.min()>=-1e-7
                row=dict(fold=fold,rep=rep,weights=d["weights"].tolist(),
                    condition=float(d["condition"]),radius=float(d["radius"]),
                    shrunk=bool(d["shrunk"]),projection_fraction=float(d["projected"].mean()),
                    outside_fraction=float(d["outside"]),min_grid_slack=float(a.min()),
                    train_rmse=float(d["reconstruction_rmse"][train].mean()),
                    train_weighted_rmse=float(d["weighted_rmse"][train].mean()))
            row["market"]=json.loads((f/rep/"market.json").read_text(encoding="utf8"))
            rows.append(row)
    write_json(out/"diagnostics.json",rows)


def conclusion(out,summary,pairs,complete,errors=()):
    rows=[r for r in summary if r["end"]==2025 and r["cost"]==5 and r["group"]=="all" and r["combo"]]
    lines=["# 50ETF 创新组合实验结果","",
           f'已完成 {complete["evaluated"]}/{complete["expected"]} 个策略训练与评价条目。按预登记规则不可运行的条目数为 {complete.get("unavailable",0)}。2024–2025 是已观察的历史评价。',"",
           "| 目标 | 组合 | 种子数 | RMSE | CVaR95 损失 | 交易成本 | 双指标未被支配 |",
           "|---|---|---:|---:|---:|---:|---|"]
    for r in rows:
        dominated=any((q["rmse"]<=r["rmse"] and q["cvar95"]<=r["cvar95"]) and
                      (q["rmse"]<r["rmse"] or q["cvar95"]<r["cvar95"])
                      for q in rows if q["lam"]==r["lam"] and q["seeds"]==3)
        lines.append(f'| {r["lam"]} | {r["combo"]} | {r["seeds"]} | {r["rmse"]:.4%} | {r["cvar95"]:.4%} | {r["trade_cost"]:.4%} | {"—" if r["seeds"]<3 else ("否" if dominated else "是")} |')
    lines += ["","差值按候选减对照计算，负值表示误差或尾部损失下降。以下列出主实验的直接机制比较，区间为三个月块配对重抽样的逐项区间。","",
              "| 目标 | 机制 | 候选 | 对照 | 指标 | 差值 | 95% 区间 |",
              "|---|---|---|---|---|---:|---:|"]
    for p in pairs:
        if (p["end"],p["cost"],p["block_months"])!=(2025,5,3):
            continue
        a,b=p["a"].split("/"),p["b"].split("/")
        tag=""
        if a[2]=="reliable_gate" and b[2] in {"static","state_gate","same_inputs","same_capacity"}:
            tag="A"
        if a[1]=="robust" and b[1]=="equal" and a[2] in {"factor","reliable_gate"}:
            tag="B"
        if a[0]=="C2" and b[0]=="C1" and a[2] in {"factor","reliable_gate"}:
            tag="C"
        if tag:
            lines.append(f'| {p["lam"]} | {tag} | {p["a"]} | {p["b"]} | {p["metric"]} | {p["delta"]:+.4%} | [{p["low"]:+.4%}, {p["high"]:+.4%}] |')
    lines += ["","逐项区间跨零时，对该指标的改善证据有限。滚动、成本和全部组合两两比较见汇总表。门控的新增训练时间和包含两个专家的从零训练时间分别列示。市场拟合与模拟不计入策略耗时，首批市场未单独记录精确耗时。",
              "","模型保留有限价格格点静态约束。实证采用同收盘交易，模拟使用无分红连续近似，真实回放计入分红。2026 年现货、期权和条款尚未核齐。"]
    lines += ["","市场内部似然用训练期尾部 10% 的转移检验固定表示下的动力学。因子表示与尺度由该折完整训练期估计。"]
    missing=defaultdict(int)
    for error in errors:
        if error.get("status")=="unavailable":
            g=error["group"]
            missing[(g["end"],g["rep"],g["market"])]+=1
    if missing:
        lines += ["","校准未通过的滚动分支："]
        lines += [f"- 评价年 {end}，{rep}，{market}，{n} 个条目。" for (end,rep,market),n in sorted(missing.items())]
    if complete.get("baseline_unavailable"):
        lines += ["",f'局部最小方差基准有 {complete["baseline_unavailable"]} 个研究环境缺失。']
    (out/"结果说明.md").write_text("\n".join(lines)+"\n",encoding="utf8")


def run(root,config):
    torch.set_num_threads(2)
    root=Path(root)
    out=root/"evaluation"
    for folder in (out,out/"daily",out/"paths"):
        folder.mkdir(exist_ok=True)
    cfg=json.loads(Path(config).read_text(encoding="utf-8-sig"))
    records=frozen_replay(root,cfg,out)
    summary,pairs=summarize(out,records,cfg)
    figures(out,summary)
    diagnostics(root,out,cfg)
    errors=json.loads((out/"failures.json").read_text(encoding="utf8"))
    frozen=json.loads((out/"frozen_checkpoints.json").read_text(encoding="utf8"))
    expected=sum(len(g["models"]) for g in tasks(cfg))
    evaluated=len(records)-sum(
        row["policy"] in {"fixed50","static","BS Delta","Local MV"}
        for row in json.loads((out/"paths_index.json").read_text(encoding="utf8")))
    unavailable=sum(e.get("status")=="unavailable" for e in errors)
    completion=dict(expected=expected,evaluated=evaluated,frozen=len(frozen),
                    unavailable=unavailable,baseline_unavailable=sum(e.get("status")=="baseline_unavailable" for e in errors),
                    failures=sum(e.get("status")=="failed" for e in errors))
    assert evaluated==len(frozen)
    conclusion(out,summary,pairs,completion,errors)
    write_json(out/"completion.json",completion)
    print("Evaluation complete:",completion,flush=True)
    if not completion["failures"] and evaluated+unavailable==expected and (root/"deliver.request").exists():
        from deliver import deliver
        project=subprocess.check_output(["git","-C",str(Path(__file__).resolve().parent),
            "rev-parse","--show-toplevel"],encoding="utf8").strip()
        deliver(root,project,publish=True)
        (root/"deliver.request").unlink()


if __name__=="__main__":
    run(*sys.argv[1:])









