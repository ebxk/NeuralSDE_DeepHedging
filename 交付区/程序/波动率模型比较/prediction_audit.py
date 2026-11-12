"""Conditional, training-internal diagnostic using retained simulator scenarios.

These reused scenarios are not independent out-of-time predictive evidence.
Sample counts are reported, never inflated to a nominal Monte Carlo budget.
"""
import json
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from study_core import CONFIG,OUT,PROJECT,HERE,SHARED_SURFACES,base_factors,bootstrap_legacy,frozen_json,write_json,digest
from artifact_guard import verify_ready,check_hashes
from linear_market import shared_summary
from volatility_surfaces import bs_call,implied_vol

def calendar_interval(first,last):
    elapsed=float((np.datetime64(last['date'])-np.datetime64(first['date'])).astype('timedelta64[D]').astype(int))/365
    remaining=float((np.datetime64(first['expiry'])-np.datetime64(last['date'])).astype('timedelta64[D]').astype(int))/365
    if elapsed<0 or remaining<0:raise ValueError('Prediction interval outside live contract')
    return elapsed,remaining

def observed_increment(first,last,remaining):
    future=max(float(last['spot'])-float(last['strike']),0.) if remaining==0 else float(last['option_close'])
    return future-float(first['option_close'])

def run(end,model):
    from pathlib import Path
    import inspect
    parents={}
    def parent(path):
        path=Path(path);parents[str(path)]=digest(path);return path
    bootstrap_legacy()
    from hedge_experiment import read_paths
    from contract_ledger import EVENTS
    from extended_curve import curve
    from stochastic_volatility import price_model
    old=PROJECT/f'工作区/数据/处理后/创新组合实验/{end}'
    cfg=json.loads(parent(old/'config.json').read_text(encoding='utf-8-sig'))
    training=read_paths(parent(old/'episode_days.csv'),[cfg['split_ends'][0]])[0]
    datafile=parent(OUT/f'inputs/{end}/sim_{model}.npz');ready=parent(datafile.with_suffix('.ready.json'))
    verify_ready(ready,datafile,CONFIG,model,end)
    data=dict(np.load(datafile,allow_pickle=False))
    common=dict(np.load(parent(old/'C1/common.npz')))
    selected=np.random.default_rng(104).integers(len(training),size=len(data['spot']))
    unique=np.unique(selected);origins=unique[np.unique(np.linspace(0,len(unique)-1,min(64,len(unique))).astype(int))]
    root=OUT/'predictions';root.mkdir(exist_ok=True)
    frozen_json(root/'design.json',dict(horizons=[1,5,20],origins=64,selection='evenly spaced origin indices',
        scenarios='reuse all matching stored simulator paths',evaluation='training_internal_only',
        minimum_scenarios=2,no_claim_of_fresh_validation=True))
    raw=dict(np.load(parent(SHARED_SURFACES)))
    observed=shared_summary(raw['calls'],np.unique(raw['grid_t']),np.unique(raw['grid_k']))
    lookup=dict(zip(raw['dates'],observed));side={}
    p=OUT/f'inputs/{end}/sim_{model}.state_sidecar.npz'
    if p.exists():side=dict(np.load(parent(p),allow_pickle=False))
    if model=='nsde':
        rep=dict(np.load(parent(base_factors(end))))
        if 'xi' not in side:raise ValueError('Fresh NSDE state sidecar required')
        scale=np.load(parent(OUT/f'markets/{end}/nsde/reference_common.npz'))['scale']
    elif model.endswith('_linear'):
        rep=dict(np.load(parent(base_factors(end) if model=='original_linear' else OUT/f'representations/{end}/{model}.npz')))
        if 'xi' not in side:side['xi']=np.load(parent(OUT/f'markets/{end}/{model}/simulation.npz'))['xi']
        scale=rep['xi'][rep['dates']<str(cfg['split_ends'][0]+1)].std(axis=0)
    if model in ('heston','sabr'):
        q=json.loads(parent(OUT/f'markets/{end}/{model}/global.json').read_text(encoding='utf8'))['model']
    tables={}
    if model=='localvol':
        from generator_export import _local_origin_surface
        from local_volatility import build_table,conditional_prices
        calendar=pd.read_csv(parent(PROJECT/'工作区/数据/原始数据/50ETF/510050_华夏上证50ETF.csv'),
            encoding='gb18030').set_index('交易日期')['收盘价(元)'].sort_index()
    results=[];excluded=[]
    for origin in origins:
        path=training[int(origin)];rows=np.flatnonzero(selected==origin)
        for h in (1,5,20):
            reason=None
            if len(path)<=h:reason='path_shorter_than_forecast'
            elif any(e[0] in {r['date'] for r in path[1:h+1]} for e in EVENTS):reason='cross_ex_dividend'
            elif len({(r['strike'],r['unit']) for r in path[:h+1]})>1:reason='contract_adjustment'
            elif len(rows)<2:reason='insufficient_stored_scenarios'
            if reason:excluded.append(dict(origin=int(origin),horizon=h,reason=reason));continue
            first,last=path[0],path[h];s0=float(first['spot']);k=float(first['strike'])
            tau=(np.datetime64(first['expiry'])-np.datetime64(first['date'])).astype(float)/365
            elapsed,remaining=calendar_interval(first,last);spots=data['spot'][rows,h]
            start_info=data['f'][rows,0,3:5];future_info=data['f'][rows,h,3:5]
            actual=np.r_[np.log(float(last['spot'])/s0),lookup[last['date']][:2]-lookup[first['date']][:2]]
            samples=np.column_stack([np.log(spots/s0),future_info-start_info])
            good=(data['f'][rows,0,5:7]>0).all(1)&(data['f'][rows,h,5:7]>0).all(1)
            measured=samples[good]
            energy=(float(np.linalg.norm(measured-actual,axis=1).mean()-.5*pdist(measured).mean())
                if len(measured)>1 and (lookup[last['date']][2:]>0).all() and (lookup[first['date']][2:]>0).all() else None)
            iv=float(implied_vol(tau,k/s0,float(first['option_close'])/s0))
            if not np.isfinite(iv) or iv<=0:
                excluded.append(dict(origin=int(origin),horizon=h,reason='invalid_initial_BS_label'));continue
            from scipy.special import ndtr
            delta=float(ndtr(np.log(s0/k)/(iv*np.sqrt(tau))+.5*iv*np.sqrt(tau)))
            if model=='gbm':
                vol=data['f'][rows,0,3]
                c0=s0*bs_call(tau,k/s0,vol);c1=spots*bs_call(remaining,k/spots,vol)
            elif model=='nsde' or model.endswith('_linear'):
                c0=s0*curve(np.full(len(rows),tau),np.full(len(rows),k/s0),side['xi'][rows,0],rep,scale)[0]
                c1=spots*curve(np.full(len(rows),remaining),k/spots,side['xi'][rows,h],rep,scale)[0]
            elif model in ('heston','sabr'):
                c0=s0*price_model(model,q,tau,k/s0,np.maximum(side['latent'][rows,0],1e-12))
                c1=spots*price_model(model,q,remaining,k/spots,np.maximum(side['latent'][rows,h],1e-12))
            else:
                day=first['date']
                if day not in tables:
                    surface,_=_local_origin_surface(day,calendar,common['warm'][rows[0]],np.unique(raw['grid_t']),parent)
                    tables[day]=build_table(surface,120)
                table=tables[day]
                c0=np.full(len(rows),s0*conditional_prices(table,0,1.,tau,[k/s0])['prices'][0])
                c1=np.array([s0*conditional_prices(table,elapsed,float(s/s0),remaining,[k/s0])['prices'][0] for s in spots])
            if remaining==0:c1=np.maximum(spots-k,0)
            residual=((c1-c0)-delta*(spots-s0))/s0
            observed_residual=(observed_increment(first,last,remaining)-
                delta*(float(last['spot'])-s0))/s0
            crps=float(np.mean(abs(residual-observed_residual))-.5*pdist(residual[:,None],metric='cityblock').mean())
            lower,upper=np.quantile(residual,[.025,.975])
            results.append(dict(origin=int(origin),horizon=h,n=len(rows),energy_score_raw_units=energy,
                energy_valid_scenarios=int(good.sum()),
                residual_crps=crps,residual_observed=observed_residual,
                covered95=bool(lower<=observed_residual<=upper),residual_status='own_model_price_increments',
                observed_value_type='economic_expiry_payoff' if remaining==0 else 'real_quote',
                expiry_quote_payoff_difference=(max(float(last['spot'])-float(last['strike']),0.)-float(last['option_close']))/s0 if remaining==0 else 0.))
    table=root/f'{end}_{model}.csv';pd.DataFrame(results).to_csv(table,index=False)
    sources={name:digest(HERE/name) for name in ['prediction_audit.py','linear_market.py',
        'volatility_surfaces.py','stochastic_volatility.py','local_volatility.py']}
    sources[inspect.getsourcefile(curve)]=digest(inspect.getsourcefile(curve));check_hashes(parents);check_hashes(sources)
    write_json(root/f'{end}_{model}.json',dict(complete=True,model=model,end=end,config=CONFIG,
        source=digest(HERE/'prediction_audit.py'),sources=sources,parents=parents,
        output_sha256={table.name:digest(table)},results=len(results),excluded=excluded,
        interpretation='conditional training-internal diagnostic; scenarios reused from policy simulator',
        limitations=['unequal finite scenario counts per origin','not independent predictive validation',
            'raw-unit multivariate energy score; do not conflate with a standardized score']))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--end',type=int,required=True);p.add_argument('--model',required=True)
    args=p.parse_args();run(args.end,args.model)
