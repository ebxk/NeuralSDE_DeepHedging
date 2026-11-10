"""Static executable controls using each expert's own factor coordinates."""
import bootstrap
import json
import numpy as np
import torch
from torch import nn
from extended_data import batch
from extended_train import predictions,restore
from extended_registry import Job
from extended_policy import _at
from bottleneck_train import inputs_for,job_for,make_model_for,evaluate
from bottleneck_core import costs
from train import expected_shortfall
from prepare_common import digest,write_json
from continuation_account import account_rollout

class AlignedBlend(nn.Module):
    def __init__(self,left,right,weight):
        super().__init__()
        self.left=left;self.right=right;self.weight=weight;self.wealth=True
    def prepare(self,data):
        other=dict(data,f=data['other_f'],history=data['other_history'],history_valid=data['other_history_valid'])
        return self.left.prepare(data),self.right.prepare(other)
    def action(self,prepared,previous,wealth):
        a=self.left.action(prepared[0],previous,wealth)[0]
        b=self.right.action(prepared[1],previous,wealth)[0]
        return torch.lerp(a,b,self.weight),None

def load_policy(project,out,task):
    job=job_for(task)
    if task['end']==2025 and task['kind'] in {'flat','real'}:
        root=project/'工作区/数据/处理后/扩展组合实验'
        model=restore(root,job)
        path=root/'policies'/f'{job.key}.pt'
    else:
        model=make_model_for(job)
        path=out/'policies'/f"{task['key']}.pt"
        model.load_state_dict(torch.load(path,weights_only=True))
    return model,dict(weights=str(path),sha256=digest(path))

def static_controls(project,out,end,seed):
    base=dict(end=end,kind='plain',seed=seed,key=f'{end}_plain_s{seed}')
    inp=inputs_for(project,base,out)
    paths,data=inp.observed(inp.folds[1])
    left,source=load_policy(project,out,base)
    reference=evaluate(left,job_for(base),data)
    results=[]
    for kind in ('flat','real'):
        other=dict(end=end,kind=kind,seed=seed,key=f'{end}_{kind}_s{seed}')
        right,right_source=load_policy(project,out,other)
        other_inputs=inputs_for(project,other,out)
        other_paths,other_data=other_inputs.observed(inp.folds[1])
        assert [p[0]['episode'] for p in paths]==[p[0]['episode'] for p in other_paths]
        for weight in np.arange(11)/10:
            blend=AlignedBlend(left,right,float(weight))
            pnl=[];c=[]
            with torch.no_grad():
                for i in range(0,len(paths),512):
                    rows=slice(i,i+512)
                    x=batch(data,rows,True);y=batch(other_data,rows,True)
                    x.update(other_f=y['f'],other_history=y['history'],other_history_valid=y['history_valid'])
                    p,r,state=account_rollout(blend,x,.0005)
                    pnl.extend(p.tolist());c.extend(state['cost'].tolist())
            pnl=np.asarray(pnl)
            results.append(dict(kind=kind,weight=float(weight),mse=float(np.mean(pnl*pnl)),
                rmse=float(np.sqrt(np.mean(pnl*pnl))),cvar=float(expected_shortfall(-pnl)),
                cost=float(np.mean(c)),left=source,right=right_source))
    return dict(end=end,seed=seed,reference=reference,controls=results)

def budget_gate(records):
    """All seeds retained; promotion is resource allocation, not a risk guarantee."""
    decisions=[]
    for epsilon,kind in ((.02,'budget02'),(.05,'budget05')):
        feasible=0;tail=[];control=[]
        for r in records:
            ref=r['reference'];candidate=r[kind]
            ok=candidate['mse'] <= (1+epsilon)**2*ref['mse'] and candidate['cost']<=1.05*ref['cost']
            feasible+=int(ok)
            controls=[ref,*r['controls'],r['lambda05'],r['lambda10']]
            eligible=[c for c in controls if c['mse']<=(1+epsilon)**2*ref['mse'] and c['cost']<=1.05*ref['cost']]
            tail.append(candidate['cvar']);control.append(min(c['cvar'] for c in eligible))
        mean_control=float(np.mean(control))
        gain=(mean_control-float(np.mean(tail)))/max(abs(mean_control),1e-12)
        decisions.append(dict(kind=kind,feasible_seeds=feasible,tail_gain=gain,promote=feasible>=2 and gain>=.02))
    return dict(promote=any(x['promote'] for x in decisions),budgets=decisions)
