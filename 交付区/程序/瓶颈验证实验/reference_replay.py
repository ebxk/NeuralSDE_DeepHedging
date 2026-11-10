"""Historical controls under the same closed-loop cost and execution rules."""
import bootstrap
from pathlib import Path
import torch
from torch import nn
from bottleneck_core import FOLDS
from extended_data import Inputs
from extended_registry import Job
from extended_train import restore
from extended_replay import sensitivities
from train import make_model
from report import bs_path
from bottleneck_report import replay_task

class LegacyAdapter(nn.Module):
    def __init__(self,model):
        super().__init__();self.model=model;self.wealth=False
    def prepare(self,data):return self.model.prepare(data['f'].transpose(0,1).contiguous())
    def action(self,prepared,previous,wealth):return self.model.action(prepared,previous),None

class FixedPolicy(nn.Module):
    def __init__(self):
        super().__init__();self.dummy=nn.Parameter(torch.zeros((),dtype=torch.float64),requires_grad=False)
        self.wealth=False
    def prepare(self,data):return data['reference_ratio'].transpose(0,1)
    def action(self,prepared,previous,wealth):return prepared,None

def references(project,out,end):
    old=project/'工作区/数据/处理后/创新组合实验'
    ext=project/'工作区/数据/处理后/扩展组合实验'
    fold=next(f for f in FOLDS if f[2]==end)
    for rep in ('C0','C1'):
        inp=Inputs(old/str(end),old/str(end)/rep,old/str(end)/f'{rep}.npz',fold)
        for seed in range(3):
            job=Job(end,rep,0,'off',False,False,5,0.,seed,'single','wide','baseline')
            model=restore(ext,job);weights=ext/'policies'/f'{job.key}.pt'
            task=dict(key=f'{end}_legacy_{rep}_s{seed}',end=end,kind=f'legacy_{rep}',seed=seed)
            yield task,(model,inp,weights)
        if rep!='C0':continue
        for seed in range(3):
            model=make_model('base')
            weights=old/str(end)/'policies'/f'C0_gbm_5bp_l0_s{seed}'/'base.pt'
            model.load_state_dict(torch.load(weights,weights_only=True))
            task=dict(key=f'{end}_legacy_GBM_s{seed}',end=end,kind='legacy_GBM',seed=seed)
            yield task,(LegacyAdapter(model),inp,weights)
        paths,data=inp.observed(end)
        ds,adj=sensitivities(paths,inp)
        for name,values in (('BS',[bs_path(p) for p in paths]),('CurveDelta',ds),
                            ('LocalMV',[d+a for d,a in zip(ds,adj)])):
            ratio=torch.zeros_like(data['active'],dtype=torch.float64)
            for i,value in enumerate(values):ratio[i,:len(value)]=torch.as_tensor(value).clamp(0,1)
            data['reference_ratio']=ratio
            task=dict(key=f'{end}_{name}',end=end,kind=name,seed=-1)
            yield task,(FixedPolicy(),inp,Path(__file__))
    if end==2025:
        from bottleneck_train import inputs_for
        from validation_controls import load_policy
        for kind in ('flat','real'):
            for seed in range(3):
                task=dict(key=f'{end}_{kind}_s{seed}',end=end,kind=kind,seed=seed)
                model,source=load_policy(project,out,task)
                yield task,(model,inputs_for(project,task,out),Path(source['weights']))

def run(project,out):
    rows=[]
    for _,_,end in FOLDS:
        for task,reference in references(project,out,end):
            rows.extend(replay_task(project,out,task,reference))
    return rows
