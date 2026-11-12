"""Train only registered, matched-initialization ordinary policies."""
import argparse
import copy
import json
import time
import numpy as np
import torch
from study_core import HERE,OUT,CONFIG,registry,digest,write_json,bootstrap_legacy
from common_policy import CommonPolicy
from artifact_guard import verify_ready

def load_data(path,center,scale):
    with np.load(path,allow_pickle=False) as f:data={k:torch.from_numpy(f[k].copy()) for k in f.files}
    data['f']=(data['f']-torch.from_numpy(center))/torch.from_numpy(scale)
    return data

def subset(data,indices):return {k:v[indices] for k,v in data.items()}

def train_one(job):
    bootstrap_legacy()
    from extended_policy import simulated_pnl,rollout
    from train import score,expected_shortfall
    torch.set_num_threads(2)
    end=job['end'];root=OUT/f'inputs/{end}';out=OUT/'policies';out.mkdir(parents=True,exist_ok=True)
    simfile=root/f"sim_{job['model']}.npz"
    if not simfile.exists():raise FileNotFoundError(f'Generator artifact not ready: {simfile}')
    ready=root/f"sim_{job['model']}.ready.json"
    verify_ready(ready,simfile,CONFIG,job['model'],end)
    fold=job['fold'];sources=[HERE/'train_common.py',HERE/'common_policy.py',root/'feature_scale.npz',
        root/'identity.json',simfile,ready,root/f'real_{fold[1]}.npz']
    from pathlib import Path
    import extended_policy
    sources.append(Path(extended_policy.__file__))
    identity=dict(job=job,config=CONFIG,parents={str(p):digest(p) for p in sources})
    initial=OUT/'initial'/f"{job['initial_key']}.pt"
    if initial.exists():identity['initial_sha256']=digest(initial)
    receipt=out/f"{job['key']}.json";weight=out/f"{job['key']}.pt"
    if receipt.exists():
        old=json.loads(receipt.read_text(encoding='utf8'))
        if old['identity']!=json.loads(json.dumps(identity)):raise ValueError('Changed training identity')
        if old['weight_sha256']!=digest(weight):raise ValueError('Weight identity mismatch')
        return old
    scale=dict(np.load(root/'feature_scale.npz',allow_pickle=False))
    sim=load_data(simfile,**scale);val=load_data(root/f'real_{fold[1]}.npz',**scale)
    torch.manual_seed(job['seed']);model=CommonPolicy(job['information']=='surface')
    initials=OUT/'initial';initials.mkdir(exist_ok=True)
    initial=initials/f"{job['initial_key']}.pt"
    if initial.exists():model.load_state_dict(torch.load(initial,weights_only=True))
    else:torch.save(model.state_dict(),initial)
    identity['initial_sha256']=digest(initial)
    eta=torch.nn.Parameter(torch.tensor(1.))
    opt=torch.optim.Adam(list(model.parameters())+([eta] if job['lam'] else []),lr=CONFIG['learning_rate'])
    rng=np.random.default_rng(1000+job['seed']);n=CONFIG['simulation_train']
    trace=[];best=float('inf');best_state=None;stale=0;start=time.monotonic()
    def predict(data):
        with torch.no_grad():
            return torch.cat([rollout(model,subset(data,slice(i,i+256)),.0005)[0]
                for i in range(0,len(data['f']),256)]).numpy()
    for epoch in range(1,CONFIG['epochs']+1):
        for _ in range(CONFIG['updates_per_epoch']):
            ix=rng.integers(n,size=CONFIG['batch_size'])
            p=simulated_pnl(model,subset(sim,ix),.0005);z=100*p
            loss=z.square().mean()
            if job['lam']:loss=loss+job['lam']*(eta+torch.relu(-z-eta).mean()/.05)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective')
            opt.zero_grad();loss.backward();opt.step()
        if epoch%CONFIG['validation_every']:continue
        p=predict(val);value=float(score(p,job['lam']))
        if not np.isfinite(value):raise FloatingPointError('Nonfinite validation')
        trace.append(dict(epoch=epoch,validation=value,rmse=float(np.sqrt(np.mean(p*p))),
            cvar=float(expected_shortfall(-p)),seconds=time.monotonic()-start))
        if value<best:
            best=value;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch;stale=0
        else:stale+=1
        write_json(out/f"{job['key']}.progress.json",dict(complete=False,trace=trace))
        print(job['key'],trace[-1],flush=True)
        if stale>=CONFIG['patience']:break
    if best_state is None:raise RuntimeError('No valid checkpoint')
    torch.save(best_state,weight)
    record=dict(status='trained_validation_selected',identity=identity,weight_sha256=digest(weight),
        best_epoch=best_epoch,validation=best,trace=trace,seconds=time.monotonic()-start,
        evaluation_complete=False)
    write_json(receipt,record);return record

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--key',required=True);args=p.parse_args()
    job=next(j for j in registry() if j['key']==args.key)
    train_one(job)
