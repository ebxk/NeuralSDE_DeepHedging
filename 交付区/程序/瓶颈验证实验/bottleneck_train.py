"""Fresh matched initialization and empirically constrained training."""
import bootstrap
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
import torch
from prepare_common import digest, write_json
from extended_data import Inputs, batch
from extended_registry import Job
from extended_train import fit, predictions, make_model_for, needs_history
from extended_policy import rollout, simulated_pnl
from train import batches, expected_shortfall
from bottleneck_core import FOLDS, budget_limits, constrained_rank, costs

def job_for(task):
    kind = task['kind']
    return Job(task['end'],'C1',int(kind=='real'),
        'anneal' if kind=='teacher' else 'real' if kind=='real' else 'off',
        kind in {'real','flat'},kind in {'real','flat'},5,
        .05 if kind=='lambda05' else .1 if kind=='lambda10' else 0.,
        task['seed'],'single','flat' if kind=='flat' else 'wide','bottleneck')

def inputs_for(project, task, out):
    end, kind = task['end'],task['kind']
    old = project/'工作区/数据/处理后/创新组合实验'
    rep = old/str(end)/'C1.npz'
    folder = (project/'工作区/数据/处理后/扩展组合实验/R1'/str(end)/'C1'
              if kind=='real' else old/str(end)/'C1')
    if kind in {'stat','hedge'}:
        folder = out/'markets'/str(end)/kind
    return Inputs(old/str(end),folder,rep,
                  next(f for f in FOLDS if f[2] == end))

def identity(project, task, cfg, inputs, initial):
    here = Path(__file__).parent
    modules = [here/'bottleneck_train.py',here/'bottleneck_core.py',here/'bootstrap.py']
    modules += [here.parent/'扩展组合实验'/f'{n}.py' for n in
                ('extended_train','extended_data','extended_policy','extended_curve')]
    return dict(task=task,config=cfg,initial=digest(initial),
                source={str(p.relative_to(here.parent)):digest(p) for p in modules},
                inputs={str(p):digest(p) for p in
                        (inputs.episode_file,inputs.directory/'common.npz',inputs.directory/'single.npz')},
                torch=torch.__version__)

def evaluate(model, job, data):
    p, r, _ = predictions(model,job,data)
    c = costs(torch.from_numpy(r),data,.0005).numpy()
    return dict(mse=float(np.mean(p*p)),rmse=float(np.sqrt(np.mean(p*p))),
                cvar=float(expected_shortfall(-p)),cost=float(c.mean()),mean=float(p.mean()))

def fit_budget(model, task, job, cfg, inputs, out):
    start = time.perf_counter()
    reference = make_model_for(job)
    ref_weights = out/'policies'/f"{task['end']}_plain_s{task['seed']}.pt"
    reference.load_state_dict(torch.load(ref_weights,weights_only=True))
    sim, ids = inputs.simulation('single')
    _, val = inputs.observed(inputs.folds[1])
    n=cfg['simulation_train']
    train = {k:v[:n] for k,v in sim.items()}
    ref_train = evaluate(reference,job,train)
    ref_val = evaluate(reference,job,val)
    epsilon = .02 if task['kind']=='budget02' else .05
    train_mse, train_cost = budget_limits(ref_train['mse'],ref_train['cost'],epsilon)
    val_mse, val_cost = budget_limits(ref_val['mse'],ref_val['cost'],epsilon)
    eta=torch.nn.Parameter(torch.tensor(1.))
    optimizer=torch.optim.Adam(list(model.parameters())+[eta],lr=.003)
    multipliers=np.zeros(2)
    rng=np.random.default_rng(job.seed)
    best=None; state=None; stale=0; trace=[]; updates=0
    for epoch in range(1,cfg['epochs']+1):
        for idx in batches(ids[:n],cfg['batch_size'],rng):
            data=batch(sim,idx,False)
            p,r,_=rollout(model,data,.0005)
            z=100*p
            c=costs(r,data,.0005).mean()
            residual=torch.stack((z.square().mean()/train_mse-1,c/max(train_cost,1e-12)-1))
            objective=eta+torch.relu(-z-eta).mean()/.05
            objective=objective+(residual*torch.as_tensor(multipliers,dtype=residual.dtype)).sum()
            if not torch.isfinite(objective):
                raise FloatingPointError('Nonfinite constrained objective')
            optimizer.zero_grad(set_to_none=True)
            objective.backward(); optimizer.step()
            multipliers=np.clip(multipliers+.01*residual.detach().numpy(),0,100)
            updates+=1
        if epoch%cfg['validation_every']:
            continue
        metrics=evaluate(model,job,val)
        rank=constrained_rank(metrics['mse']*1e4/val_mse,
                              metrics['cost']/max(val_cost,1e-12),metrics['cvar'])
        trace.append(dict(epoch=epoch,rank=rank,metrics=metrics,multipliers=multipliers.tolist(),
                          eta=float(eta.detach()),updates=updates))
        if best is None or rank<best:
            best=rank; stale=0; best_epoch=epoch
            state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else:
            stale+=1
        if stale>=cfg['patience']:
            break
    return state,dict(epoch=best_epoch,best=best,seconds=time.perf_counter()-start,
        trace=trace,updates=updates,feasible=best[0]==0,reference_train=ref_train,
        reference_validation=ref_val,budgets=dict(train_mse_z=train_mse,train_cost=train_cost,
                                               validation_mse_z=val_mse,validation_cost=val_cost),
        reference_sha256=digest(ref_weights),multiplier_hit_cap=any(
            max(row['multipliers'])>=100 for row in trace))

def train_one(project,out,cfg,task):
    torch.set_num_threads(1)
    torch.manual_seed(task['seed'])
    job=job_for(task)
    model=make_model_for(job)
    initial=out/'initial'/f"{task['end']}_{'flat' if task['kind']=='flat' else 'wide'}_s{task['seed']}.pt"
    initial.parent.mkdir(parents=True,exist_ok=True)
    # One seed owns one worker group: initialization creation is not concurrent.
    if not initial.exists():
        torch.save(model.state_dict(),initial)
    model.load_state_dict(torch.load(initial,weights_only=True))
    inputs=inputs_for(project,task,out)
    stamp=identity(project,task,cfg,inputs,initial)
    weights=out/'policies'/f"{task['key']}.pt"
    status=weights.with_suffix('.json')
    if status.exists() and weights.exists():
        previous=json.loads(status.read_text(encoding='utf8'))
        if (previous.get('status')=='complete' and previous['identity']==stamp
                and previous['weights_sha256']==digest(weights)):
            return 'cached'
        raise ValueError(f"Stale artifact preserved; explicit version required: {task['key']}")
    if task['kind']=='fullstate':
        raise NotImplementedError('Full continuation-state policy requires planner gate')
    state,stats=(fit_budget(model,task,job,cfg,inputs,out) if task['kind'].startswith('budget')
                 else fit(model,job,cfg,inputs))
    temporary=weights.with_suffix('.tmp')
    torch.save(state,temporary); temporary.replace(weights)
    model.load_state_dict(torch.load(weights,weights_only=True))
    _,validation=inputs.observed(inputs.folds[1])
    metrics=evaluate(model,job,validation)
    if not task['kind'].startswith('budget'):
        selected=metrics['mse']*1e4+job.lam*(metrics['cvar']*100)
        if abs(selected-stats['best'])>1e-6:
            raise AssertionError('Reload changed selected validation objective')
    write_json(status,dict(status='complete',identity=stamp,weights_sha256=digest(weights),
                           validation=metrics,**stats))
    return 'trained'
