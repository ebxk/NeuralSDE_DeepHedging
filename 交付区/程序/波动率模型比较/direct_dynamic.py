"""Frozen-Q stochastic/LV direct hedges on the unchanged real path ledger."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtr
from study_core import CONFIG,OUT,PROJECT,HERE,bootstrap_legacy,digest,write_json
from stochastic_volatility import q_derivatives


class BenchmarkMissing(RuntimeError):
    pass


def bs_delta(t,k,sigma):
    if t<=0 or sigma<=0:return float(1. if k<1 else 0. if k>1 else .5)
    return float(ndtr(-np.log(k)/(sigma*np.sqrt(t))+.5*sigma*np.sqrt(t)))


def stochastic_hedges(kind,model,t,k,state):
    derivatives=q_derivatives(kind,model,t,k,state)
    return {'delta':float(derivatives['delta']),'lmv':float(derivatives['lmv'])}


def select_state(day,states,calendar_positions):
    current=calendar_positions[day]
    past=[d for d,v in states.items() if d<=day and d in calendar_positions
          and 0<=current-calendar_positions[d]<=CONFIG['stale_days'] and np.isfinite(v) and v>0]
    if not past:return None,dict(fallback_source='RV20_BS',source_date=day,stale_days=None)
    origin=max(past);age=current-calendar_positions[origin]
    return float(states[origin]),dict(fallback_source='same_day' if age==0 else 'past_qualified_state',
                                    source_date=origin,stale_days=age)


def rollout(rows,decision,cost_rate,delayed=False,events=None):
    """Causal callback(row,index,held_ratio), one existing financial ledger.

    Next-close submits absolute shares at close i; close i+1 fills before its
    decision. Pending orders at expiry are cancelled, not filled and unwound.
    """
    bootstrap_legacy()
    from contract_ledger import ledger,EVENTS
    if events is None:events=EVENTS
    held=0.;pending=0.;ratios=[];raw=[];submitted=[];clips=fill_clips=0
    for i,row in enumerate(rows[:-1]):
        unit=int(row['unit'])
        if delayed:
            target=float(np.clip(pending,0,unit));fill_clips+=int(target!=pending);held=target
        action=float(decision(row,i,held/unit))
        if not np.isfinite(action):raise ValueError('nonfinite direct hedge decision')
        bounded=float(np.clip(action,0,1));clips+=int(action!=bounded)
        if delayed:
            pending=bounded*unit
        else:
            held=bounded*unit
        ratios.append(held/unit);raw.append(action);submitted.append(bounded*unit)
    # Preserve the legacy Python-double cash convention under NumPy 2 NEP50.
    book=ledger(rows,list(map(float,ratios)),float(rows[0]['option_close']),float(cost_rate),events=events)
    for i,entry in enumerate(book):
        entry.update(raw_decision=raw[i] if i<len(raw) else None,
                     executed_ratio=ratios[i] if i<len(ratios) else 0.,
                     pending_absolute_shares=submitted[i] if delayed and i<len(submitted) else 0.)
    notional=float(rows[0]['spot'])*int(rows[0]['unit'])
    diagnostics=dict(decision_clip_count=clips,pending_fill_clip_count=fill_clips,
        expiry_pending_cancelled_shares=pending if delayed else 0.,
        normalized_cost=sum(r['cost'] for r in book)/notional,
        normalized_turnover=sum(abs(r['traded'])*float(row['spot']) for r,row in zip(book,rows))/notional,
        pnl=book[-1]['wealth']/notional)
    return {'book':book,'effective_ratios':ratios,'diagnostics':diagnostics}


def _key(row):
    return (str(row['date']),str(row['expiry']),float(row['strike']),float(row['spot']))


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _calendar(parent):
    bootstrap_legacy()
    from contract_ledger import EVENTS
    source=parent(PROJECT/'工作区/数据/原始数据/50ETF/510050_华夏上证50ETF.csv')
    close=pd.read_csv(source,encoding='gb18030').set_index('交易日期')['收盘价(元)'].sort_index()
    dividends=pd.Series({event[0]:float(event[3]) for event in EVENTS}).reindex(close.index,fill_value=0)
    returns=np.log((close+dividends)/close.shift())
    return close,returns,{d:i for i,d in enumerate(close.index)}


def _warm(day,returns,positions):
    index=positions[day];warm=returns.iloc[max(0,index-19):index+1].to_numpy()
    if len(warm)!=20 or not np.isfinite(warm).all():raise ValueError('insufficient causal past20 history')
    return warm


def _decisions(end,kind,paths,parent):
    close,returns,positions=_calendar(parent)
    q=None;states=None
    if kind in ('heston','sabr'):
        directory=OUT/f'markets/{end}/{kind}'
        if not (directory/'global.json').exists() or not (directory/'states.csv').exists():
            raise BenchmarkMissing('missing frozen global Q model or causal state file')
        q=_json(parent(directory/'global.json'))['model']
        if not q.get('diagnostics',{}).get('valid',False):
            raise BenchmarkMissing('global Q numerical diagnostics failed; no model substitution permitted')
        status=directory/'status.json'
        if not status.exists() or not _json(parent(status)).get('complete',False):
            raise BenchmarkMissing('stochastic calibration status is incomplete')
        frame=pd.read_csv(parent(directory/'states.csv'))
        states=dict(zip(frame.date,frame.state))
    elif kind!='localvol':raise ValueError('kind must be heston/sabr/localvol')
    unique={_key(row):row for path in paths for row in path[:-1]}
    by_day={}
    for key,row in unique.items():by_day.setdefault(key[0],[]).append(row)
    decisions={};diagnostics={'unique_market_decisions':len(unique),'origin_fallbacks':{},'lv_tables':{}}
    for day,rows in sorted(by_day.items()):
        if kind in ('heston','sabr'):
            state,metadata=select_state(day,states,positions)
            diagnostics['origin_fallbacks'][day]=metadata
            rv=float(np.sqrt(252*np.mean(_warm(day,returns,positions)**2))) if state is None else None
            for row in rows:
                t=float((np.datetime64(row['expiry'])-np.datetime64(day)).astype(float)/365)
                k=float(row['strike'])/float(row['spot'])
                if state is None:
                    delta=bs_delta(t,k,rv);hedges=dict(delta=delta,lmv=delta)
                else:
                    # Numerical errors of a supplied qualified state are genuine
                    # failures; only unavailable/stale states get RV fallback.
                    hedges=stochastic_hedges(kind,q,t,k,state)
                decisions[_key(row)]=dict(**hedges,**metadata)
        else:
            import local_volatility as lv
            from generator_export import _local_origin_surface
            surface,metadata=_local_origin_surface(day,close,_warm(day,returns,positions),[14/365,30/365,60/365,90/365],parent)
            table=lv.build_table(surface,CONFIG['localvol_horizon_days'])
            diagnostics['origin_fallbacks'][day]=metadata
            diagnostics['lv_tables'][day]=dict(surface_hash=table['surface_hash'],diagnostics=table['diagnostics'])
            expiries={}
            for row in rows:expiries.setdefault(row['expiry'],[]).append(row)
            for expiry,group in expiries.items():
                t=float((np.datetime64(expiry)-np.datetime64(day)).astype(float)/365)
                strikes=np.array([float(r['strike'])/float(r['spot']) for r in group])
                result=lv.conditional_prices(table,0.,1.,t,strikes)
                if not result['diagnostics']['valid']:raise ValueError('current-day localvol conditional PDE failed')
                for row,delta in zip(group,result['delta']):
                    # LV has one spot factor; its instantaneous LMV equals Delta.
                    decisions[_key(row)]=dict(delta=float(delta),**metadata)
            lv._PDE_CACHE.clear()
    return decisions,diagnostics


def _groups(frame):
    bootstrap_legacy();from train import expected_shortfall
    result=[]
    for key,g in frame.groupby(['end','model','hedge','execution','bp'],sort=False):
        x=g.pnl.to_numpy();threshold=float(np.quantile(x,.05))
        result.append(dict(zip(['end','model','hedge','execution','bp'],key),paths=len(g),
            mean=float(np.mean(x)),rmse=float(np.sqrt(np.mean(x*x))),std=float(np.std(x,ddof=0)),
            p05=threshold,p95=float(np.quantile(x,.95)),cvar=float(expected_shortfall(-x)),cvar05=-float(expected_shortfall(-x)),
            mean_cost=float(g.normalized_cost.mean()),mean_turnover=float(g.normalized_turnover.mean()),
            decision_clips=int(g.decision_clip_count.sum()),pending_fill_clips=int(g.pending_fill_clip_count.sum())))
    return pd.DataFrame(result)


def run_model(end,kind):
    """Evaluate one frozen benchmark, never fit to evaluation performance."""
    bootstrap_legacy()
    from hedge_experiment import read_paths
    import contract_ledger
    root=OUT/f'direct_dynamic/{end}/{kind}';root.mkdir(parents=True,exist_ok=True);parents={}
    sources={name:digest(HERE/name) for name in ['direct_dynamic.py','stochastic_volatility.py','local_volatility.py','generator_export.py']}
    def parent(path):
        path=Path(path);parents[str(path)]=digest(path);return path
    source=parent(PROJECT/f'工作区/数据/处理后/创新组合实验/{end}/episode_days.csv')
    parent(Path(contract_ledger.__file__))
    paths=read_paths(source,[end])[0]
    ids=[int(path[0]['episode']) for path in paths]
    path_identity=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
    if not paths:raise ValueError('empty real evaluation path list')
    write_json(root/'status.json',dict(complete=False,status='running',model=kind,end=int(end),
               paths=len(paths),path_ids_sha256=path_identity,parents=parents,sources=sources))
    try:
        decisions,numerical=_decisions(end,kind,paths,parent)
    except BenchmarkMissing as error:
        receipt=dict(complete=False,status='benchmark_missing',reason=str(error),end=int(end),model=kind,
                     paths=len(paths),path_ids_sha256=path_identity,parents=parents,sources=sources)
        write_json(root/'status.json',receipt);return receipt
    except Exception as error:
        write_json(root/'status.json',dict(complete=False,status='numerical_failed',model=kind,end=int(end),
                   error_type=type(error).__name__,error=str(error),parents=parents,sources=sources))
        raise
    fields=['end','model','hedge','execution','bp','episode','date','shares','traded','cost','accrued','paid',
            'receivable','liability','cash','wealth','raw_decision','executed_ratio','pending_absolute_shares',
            'fallback_source','source_date','stale_days']
    temporary=root/'daily_ledger.csv.gz.tmp';pnl=[]
    with gzip.open(temporary,'wt',encoding='utf8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for hedge in (('delta','lmv') if kind!='localvol' else ('delta',)):
            for execution,bp in [('same_close',0),('same_close',5),('same_close',10),('next_close',5)]:
                for path in paths:
                    def action(row,i,previous):return decisions[_key(row)][hedge]
                    result=rollout(path,action,bp/10000,delayed=execution=='next_close')
                    ident=dict(end=int(end),model=kind,hedge=hedge,execution=execution,bp=bp,episode=int(path[0]['episode']))
                    used=[decisions[_key(row)] for row in path[:-1]]
                    fallback_count=sum(record['fallback_source']!='same_day' for record in used)
                    pnl.append(dict(**ident,**result['diagnostics'],fallback_decisions=fallback_count,
                                    term=path[0].get('term'),moneyness=path[0].get('moneyness')))
                    for i,entry in enumerate(result['book']):
                        metadata={key:used[i][key] if i<len(used) else None for key in ['fallback_source','source_date','stale_days']}
                        writer.writerow(dict(**ident,**entry,**metadata))
    temporary.replace(root/'daily_ledger.csv.gz')
    frame=pd.DataFrame(pnl)
    for name,data in [('pnl.csv',frame),('groups.csv',_groups(frame))]:
        temporary=root/(name+'.tmp');data.to_csv(temporary,index=False);temporary.replace(root/name)
    receipt=dict(complete=True,status='direct_dynamic_complete',model=kind,end=int(end),paths=len(paths),
                 path_ids_sha256=path_identity,parents=parents,sources=sources,numerical=numerical,
                 output_sha256={name:digest(root/name) for name in ['daily_ledger.csv.gz','pnl.csv','groups.csv']},
                 cases='0/5/10bp same-close;5bp next-close absolute shares; expiry pending cancelled',
                 risk_free_rate=CONFIG['financing_rate'],continuous_yield=CONFIG['continuous_yield'])
    write_json(root/'status.json',receipt);return receipt


def run(end=None,models=('heston','sabr','localvol')):
    ends=[int(end)] if end is not None else [fold[-1] for fold in CONFIG['folds']]
    return [run_model(year,kind) for year in ends for kind in models]


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--end',type=int)
    parser.add_argument('--model',choices=['heston','sabr','localvol'])
    args=parser.parse_args();run(args.end,(args.model,) if args.model else ('heston','sabr','localvol'))
