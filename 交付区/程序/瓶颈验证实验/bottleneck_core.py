"""Preregistered jobs and risk units for the bottleneck study."""
from itertools import product
import torch

FOLDS = ((2022,2023,2025),(2019,2020,2021),(2020,2021,2022),(2021,2022,2023))

def jobs():
    result = []
    def add(end, kind, trigger='always', **extra):
        for seed in range(3):
            result.append(dict(key=f'{end}_{kind}_s{seed}', end=end, kind=kind,
                               seed=seed, trigger=trigger, **extra))
    for _,_,end in FOLDS:
        add(end,'plain')
    add(2025,'teacher')
    for end in (2021,2022,2023):
        add(end,'flat')
        add(end,'real')
    for _,_,end in FOLDS:
        for kind in ('stat','hedge'):
            add(end,kind,'prediction' if end == 2025 else 'calibration_rolling')
        for kind in ('lambda05','lambda10','budget02','budget05'):
            add(end,kind,'risk_main' if end == 2025 else 'risk_rolling')
        add(end,'fullstate','planner_diagnostic' if end == 2025 else 'planner_rolling')
    assert len(result) == 117
    return result

def budget_limits(reference_mse_p, reference_cost, epsilon):
    return 1e4*(1+epsilon)**2*reference_mse_p, 1.05*reference_cost

def constrained_rank(mse_ratio, cost_ratio, cvar):
    violation = max(mse_ratio-1, cost_ratio-1, 0)
    return (int(violation > 0), violation, cvar)

def costs(ratios, data, fee):
    spot, unit = data['spot'], data['unit']
    target = ratios.to(spot.dtype)*unit[:,:-1]*data['active']
    target = torch.cat((target, torch.zeros_like(target[:,:1])),1)
    previous = torch.cat((torch.zeros_like(target[:,:1]),target[:,:-1]),1)
    return fee*((target-previous).abs()*spot).sum(1)/(spot[:,0]*unit[:,0])
