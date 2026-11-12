"""Frozen global pricing parameters, causal daily states and train-only P drift."""
import argparse
import json
import numpy as np
import pandas as pd
from study_core import CONFIG,OUT,PROJECT,HERE,write_json,digest
from stochastic_volatility import fit_global,filter_state,estimate_p_transitions,price_model
from volatility_surfaces import implied_vol

def run(end,kind):
    fold=next(x for x in CONFIG['folds'] if x[-1]==end)
    quotes=pd.read_csv(OUT/'data/quotes.csv');quotes=quotes[quotes.date<str(end+1)]
    train=quotes[quotes.date<str(fold[0]+1)]
    dates=np.sort(train.date.unique())
    selected=dates[np.unique(np.linspace(0,len(dates)-1,min(128,len(dates))).astype(int))]
    sample=train[train.date.isin(selected)&~train.holdout].copy()
    sample=pd.concat([g.iloc[np.unique(np.linspace(0,len(g)-1,min(12,len(g))).astype(int))]
        for _,g in sample.groupby('date')],ignore_index=True)
    proxies={}
    for day,q in train[~train.holdout].groupby('date'):
        nearest=q.iloc[np.argsort(abs(q.k.to_numpy()-1)+abs(q.ttm.to_numpy()-30/365))[:3]]
        iv=implied_vol(nearest.ttm.to_numpy(),nearest.k.to_numpy(),nearest.price.to_numpy())
        vol=float(np.nanmedian(iv));proxies[day]=max(.02,vol)**(2 if kind=='heston' else 1)
    model=fit_global(kind,sample.ttm.to_numpy(),sample.k.to_numpy(),sample.price.to_numpy(),
        sample.date.map(proxies).to_numpy())
    out=OUT/f'markets/{end}/{kind}';out.mkdir(parents=True,exist_ok=True)
    identity=dict(source=digest(HERE/'stochastic_volatility.py'),runner=digest(HERE/'fit_stochastic.py'),
        quotes=digest(OUT/'data/quotes.csv'),fold=list(fold),
        global_fit='128 evenly spaced training dates; <=12 visible strike/tenor quotes per date; 5 starts')
    write_json(out/'global.json',dict(model=model,identity=identity))
    if not model['diagnostics']['valid']:
        write_json(out/'status.json',dict(complete=False,status='global_numerical_rejected',identity=identity,
            output_sha256={'global.json':digest(out/'global.json')}))
        return
    rows=[];previous=None
    for day,q in quotes.groupby('date',sort=True):
        state=filter_state(kind,model,q.ttm.to_numpy(),q.k.to_numpy(),q.price.to_numpy(),previous)
        visible=q[~q.holdout]
        masked=filter_state(kind,model,visible.ttm.to_numpy(),visible.k.to_numpy(),
            visible.price.to_numpy(),None)
        errors=price_model(kind,model,q.ttm.to_numpy(),q.k.to_numpy(),state)-q.price.to_numpy()
        held=q[q.holdout]
        held_errors=price_model(kind,model,held.ttm.to_numpy(),held.k.to_numpy(),masked)-held.price.to_numpy()
        rows.append(dict(date=day,state=state,masked_state=masked,spot=float(q.spot.iloc[0]),
            fit_rmse=float(np.sqrt(np.mean(errors**2))),
            held_rmse=float(np.sqrt(np.mean(held_errors**2))) if len(held) else None))
        previous=state
    states=pd.DataFrame(rows);states.to_csv(out/'states.csv',index=False)
    spot=pd.read_csv(PROJECT/'工作区/数据/原始数据/50ETF/510050_华夏上证50ETF.csv',
        encoding='gb18030').set_index('交易日期')['收盘价(元)'].sort_index()
    from study_core import bootstrap_legacy
    bootstrap_legacy();from contract_ledger import EVENTS
    d=states.date.to_numpy();ix=spot.index.get_indexer(d)
    dt=np.diff(d.astype('datetime64[D]')).astype(float)/365
    valid=(d[1:]<str(fold[0]+1))&(np.diff(ix)==1)&~np.isin(d[1:],[x[0] for x in EVENTS])
    p=estimate_p_transitions(kind,states.state.to_numpy()[:-1][valid],
        states.state.to_numpy()[1:][valid],dt[valid],states.spot.to_numpy()[:-1][valid],
        states.spot.to_numpy()[1:][valid])
    write_json(out/'p_parameters.json',p)
    write_json(out/'status.json',dict(complete=True,status='calibrated',identity=identity,
        state_count=len(rows),q_diffusion_shared=True,causal_state_filter=True,
        output_sha256={name:digest(out/name) for name in ('global.json','p_parameters.json','states.csv')},
        limitation='global Q fit conditional on visible-quote ATM state proxies; no joint latent likelihood claim'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--end',type=int,required=True)
    p.add_argument('--kind',choices=['heston','sabr'],required=True);args=p.parse_args()
    run(args.end,args.kind)
