"""Actual-path parity and continuation checks, including dividends and adjustments."""
import bootstrap
from pathlib import Path
import torch
import numpy as np
from extended_policy import make_policy,rollout
from extended_data import batch,books
from bottleneck_train import inputs_for
from continuation_account import account_rollout
from prepare_common import write_json

def verify():
    torch.set_num_threads(1);torch.manual_seed(20261110)
    project=bootstrap.PROJECT;out=project/'工作区/数据/处理后/瓶颈验证实验'
    inp=inputs_for(project,dict(end=2025,kind='plain'),out)
    paths,raw=inp.observed(2025)
    model=make_policy('wide',wealth=True)
    maximum=0.;resume=0.;ledger=0.
    with torch.no_grad():
        for left in range(0,len(paths),128):
            x=batch(raw,slice(left,left+128),False)
            old=rollout(model,x,.0005)[0]
            whole,r,_=account_rollout(model,x,.0005)
            maximum=max(maximum,float((old-whole).abs().max()))
            for cut in (1,5,15,x['active'].shape[1]):
                prefix=account_rollout(model,x,.0005,stop=cut)
                tail=account_rollout(model,x,.0005,initial=prefix[2])[0]
                resume=max(resume,float((whole-tail).abs().max()))
            for delay in (False,True):
                p,r,state=account_rollout(model,x,.0005,delay=delay)
                b=books(paths[left:left+128],r.numpy(),None,5)
                z=np.array([book[-1]['wealth']/(float(path[0]['spot'])*int(path[0]['unit']))
                            for path,book in zip(paths[left:left+128],b)])
                ledger=max(ledger,float(np.max(abs(p.numpy()-z))))
                assert not state['pending'].count_nonzero()
    assert max(maximum,resume,ledger)<=1e-8
    result=dict(paths=len(paths),original_parity=maximum,continuation_error=resume,
                independent_ledger_error=ledger,delay_pending_cancelled=True)
    write_json(out/'account_verification.json',result)
    print(result)

if __name__=='__main__':verify()
