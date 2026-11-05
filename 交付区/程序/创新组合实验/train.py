"""Train preregistered policies, using the real validation fold for selection."""
import copy
import csv
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import math
import msvcrt
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

from contract_ledger import ledger
from hedge_experiment import features,read_paths,real_inputs
from policy import Blend,MLP,prepare_steps,rollout
from prepare_common import digest,identity,write_json

GATES = {"state_gate","reliable_gate","residual_gate","distance_gate","constant_gate"}
EXPERTS = ["base","factor"]
ALL = EXPERTS+["state_gate","reliable_gate","same_inputs","same_capacity"]
CORE = EXPERTS+["reliable_gate"]


def tasks(cfg):
    groups = []
    for fold in cfg["folds"]:
        end = fold[-1]
        for cost in ([5,0,10] if end==2025 else [5]):
            reps = ["C0","C1","C2"] if cost==5 else ["C0","C2"]
            markets = ["single","equal","robust"] if cost==5 else ["single","robust"]
            for lam in cfg["lambdas"]:
                for seed in cfg["seeds"]:
                    for rep in reps:
                        for market in markets:
                            models = list(ALL if (end==2025 and cost==5) else CORE)
                            if (end,cost,rep,market)==(2025,5,"C0","single"):
                                models += ["residual_gate","distance_gate","constant_gate"]
                            groups.append(dict(end=end,cost=cost,rep=rep,market=market,
                                               lam=lam,seed=seed,models=models))
                    groups.append(dict(end=end,cost=cost,rep="C0",market="gbm",
                                       lam=lam,seed=seed,models=["base"]))
    return groups


def group_path(root,group):
    g=group
    return Path(root)/str(g["end"])/"policies"/f'{g["rep"]}_{g["market"]}_{g["cost"]}bp_l{g["lam"]}_s{g["seed"]}'


def make_model(name,experts=None,fixed=None):
    if name in GATES or fixed is not None:
        return Blend(copy.deepcopy(experts["base"]),copy.deepcopy(experts["factor"]),name,fixed)
    return MLP({"base":3,"factor":5,"same_inputs":8,"same_capacity":8}[name],
               width=46 if name=="same_capacity" else 32)


def expected_shortfall(loss, alpha=.95, axis=-1):
    loss=np.sort(np.asarray(loss),axis=axis)
    k=(1-alpha)*loss.shape[axis]
    indices=np.arange(loss.shape[axis]-1,loss.shape[axis]-int(np.ceil(k))-1,-1)
    tail=np.take(loss,indices,axis=axis)
    weight=np.clip(k-np.arange(len(indices)),0,1)
    return np.sum(np.moveaxis(tail,axis,-1)*weight,axis=-1)/k


def score(pnl,lam):
    z=np.asarray(pnl)*100
    return float(np.mean(z*z)+lam*expected_shortfall(-z))


def adversary(risk,radius=.1):
    risk=np.asarray(risk,dtype=float)
    m=len(risk)
    if not radius or m==1 or np.ptp(risk)<1e-12:
        return np.full(m,1/m)
    z=(risk-risk.max())/np.ptp(risk)
    lo,hi=0.,2048.
    for _ in range(32):
        k=(lo+hi)/2
        p=np.exp(k*z)
        p/=p.sum()
        kl=np.sum(p*(k*z-np.log(np.exp(k*z).sum())+math.log(m)))
        if kl>radius:
            hi=k
        else:
            lo=k
    p=np.exp(lo*z)
    return p/p.sum()


def pnl_sim(policy,f,prices,active,terminal,fee):
    pnl=terminal
    previous=torch.zeros_like(pnl)
    for step,prepared in enumerate(prepare_steps(policy,f)):
        target=policy.action(prepared,previous)
        target=torch.where(active[:,step],target,previous)
        pnl=pnl+target*(prices[:,step+1]-prices[:,step])-fee*(target-previous).abs()*prices[:,step]
        previous=target
    return pnl-fee*previous*prices[:,-1]


def make_real(paths,common,episode_file):
    f,u=real_inputs(paths,common,True)
    offsets={}
    with Path(episode_file).open(encoding="utf8",newline="") as file:
        for i,row in enumerate(csv.DictReader(file)):
            offsets.setdefault(row["episode"],i)
    quality=np.zeros((*f.shape[:2],3),dtype="f")
    for i,p in enumerate(paths):
        j=offsets[p[0]["episode"]]
        quality[i,:len(p)-1]=common["real_q"][j:j+len(p)-1]
    return torch.cat([f,torch.from_numpy(quality)],axis=2),u


def replay(model,paths,inputs,fee,capture=False):
    with torch.no_grad():
        ratios,gate=rollout(model,*inputs,capture_gate=capture)
    ratios=ratios.numpy()
    gate=None if gate is None else gate.numpy()
    pnl=np.empty(len(paths))
    books=[] if capture else None
    for i,path in enumerate(paths):
        book=ledger(path,ratios[i,:len(path)-1],float(path[0]["option_close"]),fee)
        pnl[i]=book[-1]["wealth"]/(float(path[0]["spot"])*int(path[0]["unit"]))
        if capture:
            for j,row in enumerate(book):
                row["gate_weight"]=float(gate[i,j]) if gate is not None and j<len(book)-1 else ""
            books.append(book)
    return pnl,books


def make_sim(directory,market):
    common=dict(np.load(directory/"common.npz"))
    if market=="gbm":
        spot=common["gbm"]
        other=None
        ids=np.zeros(len(spot),dtype=np.int8)
    else:
        with np.load(directory/("single.npz" if market=="single" else "pool.npz")) as f:
            spot=f["spot"]
            other=np.concatenate([f["xi"],f["quality"]],axis=2)
            ids=f["member"]
    t=common["time"]
    changes=np.concatenate([common["warm"],np.diff(np.log(spot),axis=1)],axis=1)
    rv=np.sqrt(252*np.mean(np.lib.stride_tricks.sliding_window_view(changes,20,axis=1)**2,axis=2))
    strike=common["strike"]
    f=features(spot,strike[:,None],t[:,-1:]-t,rv,other)[:,:-1]
    f,prices,active,terminal=map(torch.from_numpy,(f.copy(),(spot/spot[:,:1]).astype("f"),
        np.diff(t,axis=1)>0,((common["premium"]-np.maximum(spot[:,-1]-strike,0))/spot[:,0]).astype("f")))
    return (f,prices,active,terminal),ids,common


def batches(ids,batch,rng):
    shuffled=[rng.permutation(np.flatnonzero(ids==j)) for j in np.unique(ids)]
    n=math.ceil(len(ids)/batch)
    pieces=[np.array_split(x,n) for x in shuffled]
    for i in range(n):
        yield np.concatenate([x[i] for x in pieces])


def train_one(model,g,cfg,sim,ids,validation,inputs,weights,status,identity):
    start=time.perf_counter()
    n=cfg["simulation_train"]
    fee,lam=g["cost"]/10000,g["lam"]
    eta=torch.nn.Parameter(torch.tensor(1.))
    params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.Adam(params+([eta] if lam else []),lr=.001 if isinstance(model,Blend) else .003)
    rng=np.random.default_rng(g["seed"])
    best,stale=float("inf"),0
    trace=[]
    for epoch in range(1,cfg["epochs"]+1):
        totals=[]
        for idx in batches(ids[:n],cfg["batch_size"],rng):
            opt.zero_grad()
            z=100*pnl_sim(model,*(x[idx] for x in sim),fee)
            loss=z.square()
            if lam:
                loss=loss+lam*(eta+torch.relu(-z-eta)/.05)
            levels=np.unique(ids[idx])
            risks=torch.stack([loss[ids[idx]==i].mean() for i in levels])
            q=adversary(risks.detach().numpy(),.1 if g["market"]=="robust" else 0)
            (risks*torch.as_tensor(q,dtype=risks.dtype)).sum().backward()
            opt.step()
            totals.append(np.r_[risks.detach().numpy(),q])
        if epoch%cfg["validation_every"]:
            continue
        pnl,_=replay(model,validation,inputs,fee)
        value=score(pnl,lam)
        if not np.isfinite(value) or not np.isfinite(totals).all():
            raise FloatingPointError("Non-finite training or validation risk")
        with torch.no_grad():
            zp=100*pnl_sim(model,*(x[n:] for x in sim),fee)
            loss=zp.square()+(lam*(eta+torch.relu(-zp-eta)/.05) if lam else 0)
            risks=np.array([loss[ids[n:]==i].mean().item() for i in levels])
            qv=adversary(risks,.1 if g["market"]=="robust" else 0)
        m=len(levels)
        train_stats=np.mean(totals,axis=0)
        trace.append(dict(epoch=epoch,validation=value,rmse=float(np.sqrt(np.mean(pnl**2))),
            cvar=float(expected_shortfall(-pnl)),eta=float(eta.detach()),
            train_risk=train_stats[:m].tolist(),train_q=train_stats[m:].tolist(),
            sim_risk=risks.tolist(),sim_q=qv.tolist(),
            sim_kl=float(np.sum(qv*np.log(np.maximum(qv*m,1e-300)))),
            effective_markets=float(1/np.sum(qv*qv))))
        if value<best:
            best,stale,best_epoch=value,0,epoch
            checkpoint=copy.deepcopy(model.state_dict())
        else:
            stale+=1
        if stale>=cfg["patience"]:
            break
    tmp=weights.with_suffix(".tmp")
    torch.save(checkpoint,tmp)
    tmp.replace(weights)
    model.load_state_dict(torch.load(weights,weights_only=True))
    assert abs(score(replay(model,validation,inputs,fee)[0],lam)-best)<1e-10
    write_json(status,dict(status="complete",identity=identity,epoch=best_epoch,best=best,
        seconds=time.perf_counter()-start,parameters=sum(p.numel() for p in model.parameters()),
        weights_sha256=digest(weights),trace=trace))


def group_run(root,g,cfg,token,core_only=False):
    folder=group_path(root,g)
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'.train.lock').open('a+b') as lock:
        lock.seek(0)
        while True:
            try:
                msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
                break
            except OSError:
                time.sleep(2)
        try:
            return train_group(root,g,cfg,token,core_only)
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)


def train_group(root,g,cfg,token,core_only=False):
    torch.set_num_threads(1)
    folder=Path(root)/str(g["end"])
    directory=folder/g["rep"]
    out=group_path(root,g)
    out.mkdir(parents=True,exist_ok=True)
    successes={}
    expert_hashes={}
    failure=None
    try:
        manifest=directory/"market.done.json"
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        sim,ids,common=make_sim(directory,g["market"])
        fold=next(f for f in cfg["folds"] if f[-1]==g["end"])
        validation=read_paths(folder/"episode_days.csv",[fold[1]])[0]
        inputs=make_real(validation,common,folder/"episode_days.csv")
        source=dict(token=token,group=g,parent=digest(manifest),episodes=digest(folder/"episodes.done.json"))
    except Exception:
        failure=traceback.format_exc()
    for name in (CORE if core_only else g["models"]):
        status=out/f"{name}.json"
        weights=out/f"{name}.pt"
        try:
            if failure:
                raise RuntimeError(failure)
            torch.manual_seed(g["seed"])
            model=make_model(name,successes)
            model_source={**source,"experts":dict(expert_hashes) if name in GATES else {}}
            if status.exists() and weights.exists():
                record=json.loads(status.read_text(encoding="utf8"))
            else:
                record={}
            if (record.get("status")=="complete" and record["identity"]==model_source
                and digest(weights)==record["weights_sha256"]):
                model.load_state_dict(torch.load(weights,weights_only=True))
            else:
                train_one(model,g,cfg,sim,ids,validation,inputs,weights,status,model_source)
            if name in EXPERTS:
                successes[name]=model
                expert_hashes[name]=digest(weights)
        except Exception:
            write_json(status,dict(status="failed",reason=traceback.format_exc()))
    if set(successes)==set(EXPERTS):
        candidates=np.linspace(0,1,11)
        scores=[score(replay(make_model("fixed",successes,float(w)),validation,inputs,
                       g["cost"]/10000)[0],g["lam"]) for w in candidates]
        write_json(out/"static.json",dict(identity={**source,"experts":expert_hashes},weight=float(candidates[np.argmin(scores)]),scores=scores))
    return g


def verify_legacy(root,old):
    torch.set_num_threads(1)
    old=Path(old)
    cfg=json.loads((Path(root)/"2025"/"config.json").read_text(encoding="utf8"))
    paths=read_paths(old/"episode_days.csv",[cfg["split_ends"][-1]])[0]
    sim=np.load(old/"joint_paths.npz")
    actual={}
    with (old/"dh_ledger.csv").open(encoding="utf8",newline="") as file:
        for row in csv.DictReader(file):
            if row["date"]==row["expiry"]:
                actual[row["group"],int(row["bp"]),int(row["seed"]),row["episode"]]=float(row["wealth"])/float(row["initial_notional"])
    worst=0.
    checked=0
    for file in (old/"hedge_jobs").glob("*.pt"):
        # The original jobs' status files carry unambiguous group/cost/seed metadata.
        status=file.with_suffix(".json")
        if not status.exists():
            continue
        meta=json.loads(status.read_text(encoding="utf8"))
        name,bp,seed=meta["group"],meta["bp"],meta["seed"]
        has=name.endswith("factors")
        model=MLP(5 if has else 3)
        model.load_state_dict(torch.load(file,weights_only=True))
        f,u=real_inputs(paths,sim,has)
        pnl,_=replay(model,paths,(f,u),bp/10000)
        target=np.array([actual[name,bp,seed,p[0]["episode"]] for p in paths])
        worst=max(worst,float(np.max(np.abs(pnl-target))))
        checked+=1
    assert checked==18 and worst<=1e-8,(checked,worst)
    return worst


def run(root,config,old):
    cfg=json.loads(Path(config).read_text(encoding="utf-8-sig"))
    groups=tasks(cfg)
    assert sum(len(g["models"]) for g in groups)==1008
    here=Path(__file__).resolve().parent
    token=identity([here/"train.py",here/"policy.py",config],dict(torch=torch.__version__))
    root=Path(root)
    write_json(root/"registry.json",dict(groups=groups,source=token))
    worst=verify_legacy(root,old)
    write_json(root/"legacy_replay.json",dict(max_error=worst))
    with ProcessPoolExecutor(max_workers=int(os.environ.get("STUDY_WORKERS","6"))) as pool:
        running={pool.submit(group_run,root,g,cfg,token):g for g in groups}
        for i,future in enumerate(as_completed(running),1):
            g=running[future]
            try:
                future.result()
            except Exception:
                write_json(group_path(root,g)/"worker_failure.json",dict(error=traceback.format_exc()))
            print(f"{i}/{len(groups)}",g,flush=True)




if __name__=="__main__":
    run(*sys.argv[1:])




