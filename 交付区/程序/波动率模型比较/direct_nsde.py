"""Reload the corrected predecessor NSDE sensitivities, preserving actual shares."""
import argparse
import json
import numpy as np
import pandas as pd
from study_core import OUT,PROJECT,base_factors,bootstrap_legacy,digest,write_json

def run(end):
    bootstrap_legacy()
    from extended_data import Inputs
    from extended_replay import sensitivities
    from contract_ledger import ledger
    old=PROJECT/f'工作区/数据/处理后/创新组合实验/{end}'
    cfg=json.loads((old/'config.json').read_text(encoding='utf-8-sig'))
    inp=Inputs(old,old/'C1',base_factors(end),cfg['split_ends'])
    fresh=OUT/f'markets/{end}/nsde/reference_common.npz'
    with np.load(fresh,allow_pickle=False) as f:
        for key in ('real_xi','real_cov','scale','real_dates'):inp.common[key]=f[key]
    paths=inp.paths[end];deltas,corrections=sensitivities(paths,inp)
    rows=[];daily=[];decisions=[]
    for name in ('curve_delta','local_minimum_variance'):
        for cost,execution in ((0,'same_close'),(5,'same_close'),(10,'same_close'),(5,'next_close')):
            for path,d,c in zip(paths,deltas,corrections):
                raw=d if name=='curve_delta' else d+c
                ratios=list(map(float,np.clip(raw,0,1)))
                if execution=='next_close':
                    # The order is ABSOLUTE shares from the prior decision, not a shifted ratio.
                    ratios=[0.]+[ratios[j-1]*int(path[j-1]['unit'])/int(path[j]['unit']) for j in range(1,len(ratios))]
                book=ledger(path,ratios,float(path[0]['option_close']),cost/10000)
                nominal=float(path[0]['spot'])*int(path[0]['unit'])
                ident=dict(method=name,cost_bp=cost,execution=execution,episode=int(path[0]['episode']))
                rows.append(dict(**ident,pnl=book[-1]['wealth']/nominal,
                    cost=sum(x['cost'] for x in book)/nominal,
                    turnover=sum(abs(x['traded'])*float(y['spot']) for x,y in zip(book,path))/nominal,
                    clip_decisions=int(np.count_nonzero((raw<0)|(raw>1)))))
                daily.extend(dict(**ident,**r) for r in book)
                if cost==5 and execution=='same_close':
                    decisions.extend(dict(**ident,date=r['date'],curve_delta=float(a),covariance_correction=float(b),
                        unclipped_target=float(v),clipped_target=float(np.clip(v,0,1)))
                        for r,a,b,v in zip(path,d,c,raw))
    root=OUT/f'direct_nsde/{end}';root.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(root/'pnl.csv',index=False)
    pd.DataFrame(daily).to_csv(root/'daily_ledger.csv.gz',index=False)
    pd.DataFrame(decisions).to_csv(root/'sensitivity_components.csv.gz',index=False)
    write_json(root/'status.json',dict(complete=True,paths=len(paths),
        factors_sha256=digest(base_factors(end)),common_sha256=digest(old/'C1/common.npz'),
        fitted_reference_sha256=digest(fresh),
        episodes_sha256=digest(old/'episode_days.csv'),delta_and_covariance_correction_separated=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--end',type=int,required=True);run(p.parse_args().end)
