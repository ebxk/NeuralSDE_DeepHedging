"""Interim frozen main-period replay requested during long rolling training."""
import csv
import json
from pathlib import Path
from dataclasses import asdict
import time

import numpy as np
import torch

from prepare_common import digest,write_json
from extended_registry import registry
from extended_data import Inputs
from extended_train import predictions,restore,state_file
from extended_report import boot_metrics
from run_extended import sources
from train import expected_shortfall


def run(project):
    torch.set_num_threads(1)
    here=Path(__file__).resolve().parent
    root=project/"工作区/数据/处理后/扩展组合实验"
    cfg=json.loads((here/"study.json").read_text(encoding="utf8"))
    selected=[j for j in registry(cfg).values() if j.family=="main" and j.end==2025 and j.cost==5 and j.lam==0]
    assert len(selected)==144
    frozen={}
    for j in selected:
        path=state_file(root,j)
        r=json.loads(path.with_suffix(".json").read_text(encoding="utf8"))
        assert r["status"]=="complete" and digest(path)==r["weights_sha256"]
        frozen[j.key]=dict(job=j.meta(),sha256=digest(path),best_epoch=r["epoch"])
    out=root/"interim_20261108"
    out.mkdir(exist_ok=True)
    write_json(out/"frozen_checkpoints.json",frozen)
    groups={}
    ids=None
    months=None
    for r in (0,1):
        for rep in ("C0","C1","C2"):
            paths,_=sources(project,root,(2025,r,rep,"single"))
            inputs=Inputs(*paths,cfg["folds"][0])
            test,data=inputs.observed(2025)
            current=[p[0]["episode"] for p in test]
            assert len(test)==773 and (ids is None or ids==current)
            ids=current
            months=np.array([p[0]["expiry"][:7] for p in test])
            for j in selected:
                if (j.r,j.rep)!=(r,rep):continue
                assert digest(state_file(root,j))==frozen[j.key]["sha256"]
                pnl,_,_=predictions(restore(root,j),j,data)
                key=f"{j.rep}_R{j.r}_T{j.t}_W{int(j.w)}H{int(j.h)}"
                groups.setdefault(key,{})[j.seed]=pnl
            print("Replayed",r,rep,flush=True)
    values={k:np.stack([s[i] for i in (0,1,2)]) for k,s in groups.items()}
    np.savez_compressed(out/"path_pnl.npz",episode=np.array(ids),month=months,**values)
    points={k:np.stack([np.sqrt(np.mean(x*x,axis=1)),expected_shortfall(-x,axis=1)],axis=1) for k,x in values.items()}
    reference="C1_R0_Toff_W0H0"
    ref=points[reference].mean(0)
    draws={}
    for k,x in values.items():
        # Only the first two bootstrap columns are used. Cost/turnover are not estimated here.
        placeholder=np.stack([x,np.zeros_like(x),np.zeros_like(x)],axis=-1)
        draws[k]=boot_metrics(placeholder,months,3,2000)[:,:2]
    rows=[]
    for k,p in points.items():
        delta=p.mean(0)-ref
        lo,hi=np.quantile(draws[k]-draws[reference],[.025,.975],axis=0)
        row=dict(method=k,seeds=3,n=773,rmse=float(p[:,0].mean()),cvar95=float(p[:,1].mean()),
                 rmse_min=float(p[:,0].min()),rmse_max=float(p[:,0].max()),
                 cvar_min=float(p[:,1].min()),cvar_max=float(p[:,1].max()),
                 rmse_delta=float(delta[0]),cvar_delta=float(delta[1]),
                 rmse_low=float(lo[0]),rmse_high=float(hi[0]),cvar_low=float(lo[1]),cvar_high=float(hi[1]),
                 better_rmse_seeds=int(np.sum(p[:,0]<points[reference][:,0])),
                 better_cvar_seeds=int(np.sum(p[:,1]<points[reference][:,1])))
        rows.append(row)
    rows.sort(key=lambda row:row["rmse"])
    with (out/"summary_5bp_mse.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    write_json(out/"status.json",dict(stage="interim",models=144,configurations=48,cost_bp=5,lam=0,
               paths=773,seeds=[0,1,2],reference=reference,historical_evaluation="2024-2025 already observed",
               checkpoint_selection="unchanged validation selection",bootstrap=2000,block_months=3))
    for row in rows[:8]:print(json.dumps(row),flush=True)
    for k in (reference,"C1_R1_Tanneal_W1H1"):
        print("REFERENCE_OR_FULL",json.dumps(next(row for row in rows if row["method"]==k)),flush=True)


if __name__=="__main__":
    run(Path.cwd())
