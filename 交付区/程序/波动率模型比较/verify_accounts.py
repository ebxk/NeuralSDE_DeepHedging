"""Independent all-path ledger and unchanged legacy-weight replay checks."""
import numpy as np
import torch
from study_core import OUT,PROJECT,bootstrap_legacy,write_json,digest
from common_policy import CommonPolicy
from train_common import load_data,subset
from delayed_account import delayed_rollout

def run():
    bootstrap_legacy()
    from extended_policy import rollout,make_policy,load_legacy
    from extended_data import Inputs
    from train import make_model,replay,make_real
    from hedge_experiment import read_paths
    from contract_ledger import ledger
    torch.set_num_threads(1);torch.manual_seed(20261112)
    root=OUT/'inputs/2025';old=PROJECT/'工作区/数据/处理后/创新组合实验/2025'
    paths=read_paths(old/'episode_days.csv',[2025])[0]
    x=load_data(root/'real_2025.npz',**dict(np.load(root/'feature_scale.npz')))
    model=CommonPolicy(True);maximum=0.
    with torch.no_grad():
        for start in range(0,len(paths),128):
            data=subset(x,slice(start,start+128))
            for delayed in (False,True):
                p,r=(delayed_rollout(model,data,.0005) if delayed else rollout(model,data,.0005)[:2])
                for i,path in enumerate(paths[start:start+128]):
                    book=ledger(path,map(float,r[i,:len(path)-1].numpy()),float(path[0]['option_close']),.0005)
                    z=book[-1]['wealth']/(float(path[0]['spot'])*int(path[0]['unit']))
                    maximum=max(maximum,abs(z-float(p[i])))
    inp=Inputs(old,old/'C1',old/'C1.npz',[2022,2023,2025])
    paths,raw=inp.observed(2025)
    weight=old/'policies/C1_single_5bp_l0_s0/same_capacity.pt'
    state=torch.load(weight,weights_only=True)
    before=make_model('same_capacity');before.load_state_dict(state)
    migrated=make_policy('wide',wealth=False,history=False);load_legacy(migrated,state)
    # NumPy 2 scalar promotion changes np.float32 * Python-int ledger arithmetic.
    # Restore the predecessor's Python-float ledger convention without editing it.
    original_ledger=replay.__globals__['ledger']
    replay.__globals__['ledger']=lambda path,ratios,premium,cost:original_ledger(path,map(float,ratios),premium,cost)
    try:p0=replay(before,paths,make_real(paths,inp.common,inp.episode_file),.0005)[0]
    finally:replay.__globals__['ledger']=original_ledger
    with torch.no_grad():
        p1=torch.cat([rollout(migrated,subset(raw,slice(i,i+128)),.0005)[0]
            for i in range(0,len(paths),128)]).numpy()
    legacy=float(np.max(abs(p0-p1)))
    if max(maximum,legacy)>1e-8:raise AssertionError(f'Account mismatch independent={maximum}, legacy={legacy}')
    write_json(OUT/'account_acceptance.json',dict(complete=True,paths=len(paths),
        independent_ledger_error=maximum,legacy_reload_error=legacy,legacy_weight_sha256=digest(weight),
        includes_dividends_adjustments_and_next_close=True))

if __name__=='__main__':run()
