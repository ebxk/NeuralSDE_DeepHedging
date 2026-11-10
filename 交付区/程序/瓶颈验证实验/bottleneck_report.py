"""Rebuild ledgers under execution stresses after validation choices freeze."""
import bootstrap
import gzip
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from bottleneck_train import inputs_for,job_for,make_model_for
from continuation_account import account_rollout
from bottleneck_core import costs
from contract_ledger import EVENTS
from prepare_common import digest,write_json
from extended_data import books,batch
from train import expected_shortfall

def replay_task(project,out,task,reference=None):
    key=task['key'];weights=out/'policies'/f'{key}.pt'
    if reference is None:
        record=json.loads(weights.with_suffix('.json').read_text(encoding='utf8'))
        assert record['status']=='complete' and record['weights_sha256']==digest(weights)
        inp=inputs_for(project,task,out);job=job_for(task)
        model=make_model_for(job);model.load_state_dict(torch.load(weights,weights_only=True))
    else:
        model,inp,weights=reference
        record=(json.loads(weights.with_suffix('.json').read_text(encoding='utf8'))
                if weights.suffix=='.pt' and weights.with_suffix('.json').exists() else {})
        job=job_for(task)
    paths,data=inp.observed(task['end'])
    rows=[]
    for bp in (0,5,10):
        for delay in (False,True):
            started=time.perf_counter()
            pnl=[];ratios=[];fees=[]
            with torch.no_grad():
                for left in range(0,len(paths),512):
                    part=batch(data,slice(left,left+512),job.h or job.policy=='flat')
                    p,r,s=account_rollout(model,part,bp/10000,delay=delay)
                    pnl.extend(p.tolist());ratios.extend(r.tolist());fees.extend(s['cost'].tolist())
            pnl=np.asarray(pnl);ratios=np.asarray(ratios)
            turnover=costs(torch.from_numpy(ratios),data,1.).numpy()
            ledgers=books(paths,ratios,None,bp)
            independent=np.array([b[-1]['wealth']/(float(p[0]['spot'])*int(p[0]['unit'])) for p,b in zip(paths,ledgers)])
            error=float(np.max(abs(pnl-independent)))
            if error>1e-8:raise AssertionError(f'Independent ledger mismatch {key}: {error}')
            name=f'{key}_{bp}bp_delay{int(delay)}'
            folder=out/'evaluation';(folder/'daily').mkdir(parents=True,exist_ok=True)
            (folder/'paths').mkdir(exist_ok=True)
            daily=folder/'daily'/f'{name}.csv.gz'
            with gzip.open(daily,'wt',encoding='utf8',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(ledgers[0][0]))
                writer.writeheader()
                for ledger in ledgers:writer.writerows(ledger)
            row=dict(key=key,end=task['end'],kind=task['kind'],seed=task['seed'],cost_bp=bp,delay=delay,
                n=len(paths),rmse=float(np.sqrt(np.mean(pnl*pnl))),cvar=float(expected_shortfall(-pnl)),
                mean=float(pnl.mean()),std=float(pnl.std()),cost=float(np.mean(fees)),
                turnover=float(turnover.mean()),ledger_error=error,
                parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                training_seconds=record.get('seconds',0),replay_seconds=time.perf_counter()-started)
            write_json(folder/'paths'/f'{name}.json',dict(metrics=row,weights_sha256=digest(weights),
                daily_sha256=digest(daily),rows=[dict(episode=p[0]['episode'],pnl=float(z),
                    expiry=p[0]['expiry'],term=p[0]['term'],moneyness=p[0]['moneyness'],
                    adjusted=any(int(r['unit'])!=10000 for r in p),cost=float(f),turnover=float(v),
                    cross_dividend=any(p[0]['date']<event[0]<=p[-1]['date'] for event in EVENTS))
                    for p,z,f,v in zip(paths,pnl,fees,turnover)]))
            rows.append(row)
    return rows
