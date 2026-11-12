"""Exact existing real ledger, distinct Black and frozen-parameter smile deltas."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtr
from study_core import OUT,PROJECT,HERE,CONFIG,bootstrap_legacy,write_json,digest
from volatility_surfaces import price_surface,implied_vol

def hedge_delta(model,t,k,kind):
    price=float(price_surface(model,t,k))
    if kind=='black':
        iv=float(implied_vol(t,k,price))
        if not np.isfinite(iv) or iv<=0:raise ValueError('Unidentifiable IV')
        return float(ndtr(-np.log(k)/(iv*np.sqrt(t))+.5*iv*np.sqrt(t)))
    eps=1e-4
    # C(S,K)=S*c(K/S); holding the entire model parameterization fixed.
    up=(1+eps)*float(price_surface(model,t,k/(1+eps)))
    down=(1-eps)*float(price_surface(model,t,k/(1-eps)))
    return (up-down)/(2*eps)

def rv20_series(close,events):
    close=close.sort_index()
    dividends=pd.Series({x[0]:float(x[3]) for x in events}).reindex(close.index,fill_value=0)
    returns=np.log((close+dividends)/close.shift())
    return np.sqrt(252*returns.pow(2).rolling(20,min_periods=20).mean())


def surface_lookup(family,used_dates,days,parent):
    positions={str(d):i for i,d in enumerate(days)};wanted=set(used_dates);cache={};last=None
    first=max(0,positions[min(used_dates)]-CONFIG['stale_days']);final=positions[max(used_dates)]
    # Include all trading days and the five-day prefix before evaluation begins.
    for day in days[first:final+1]:
        day=str(day);path=OUT/f'surfaces/{family}/{day}/full.json'
        if path.exists():
            record=json.loads(Path(parent(path)).read_text(encoding='utf8'))
            if record.get('status')=='complete' and record['model'].get('diagnostics',{}).get('valid',False):
                last=(day,record['model'])
        if day in wanted:
            qualified=last is not None and positions[day]-positions[last[0]]<=CONFIG['stale_days']
            cache[day]=dict(model=last[1] if qualified else None,source_date=last[0] if qualified else day,
                           stale_days=positions[day]-positions[last[0]] if qualified else None,
                           fallback_source=('same_day' if day==last[0] else 'past_qualified_curve') if qualified else 'RV20_BS')
    return cache


def execute_path(rows,ratios,cost_bp,execution,events=None):
    from direct_dynamic import rollout
    ratios=list(map(float,ratios))
    if execution not in ('same_close','next_close'):raise ValueError('unknown execution convention')
    return rollout(rows,lambda row,i,previous:ratios[i],float(cost_bp)/10000,
                   delayed=execution=='next_close',events=events)


def _run(end,root,parents):
    bootstrap_legacy()
    from hedge_experiment import read_paths
    import contract_ledger
    import bs_baseline
    from bs_baseline import delta
    def parent(path):
        path=Path(path);parents[str(path)]=digest(path);return path
    stage=json.loads(parent(OUT/'surface_stage.json').read_text(encoding='utf8'))
    if not stage['stage_complete']:raise ValueError('Full surface stage is not yet finished')
    source=parent(PROJECT/f'工作区/数据/处理后/创新组合实验/{end}/episode_days.csv')
    parent(contract_ledger.__file__);parent(bs_baseline.__file__)
    paths=read_paths(source,[end])[0]
    if not paths:raise ValueError('no real evaluation paths')
    calendar=pd.read_csv(parent(PROJECT/'工作区/数据/原始数据/50ETF/510050_华夏上证50ETF.csv'),encoding='gb18030')
    close=calendar.set_index('交易日期')['收盘价(元)'].sort_index();days=close.index.to_numpy()
    rv=rv20_series(close,contract_ledger.EVENTS)
    used_dates=sorted({r['date'] for p in paths for r in p[:-1]})
    for day in used_dates:
        if not np.isfinite(rv.loc[day]):raise ValueError('insufficient causal RV20 warm history: '+day)
    def key(row):return (row['date'],row['expiry'],float(row['strike']),float(row['spot']),float(row['option_close']))
    unique={key(row):row for path in paths for row in path[:-1]};actions={};metadata={}
    for family in ('bs','svi','ssvi','essvi'):
        cache=surface_lookup(family,used_dates,days,parent) if family!='bs' else None
        for kind in (('black',) if family=='bs' else ('black','smile')):
            for ident,row in unique.items():
                day=row['date'];vol=max(float(rv.loc[day]),.01)
                meta=dict(fallback_source='current_quote',source_date=day,stale_days=0)
                if family=='bs':
                    ratio,_,fallback=delta(row,vol)
                    if fallback:meta['fallback_source']='RV20_BS'
                else:
                    state=cache[day];meta={k:state[k] for k in ('fallback_source','source_date','stale_days')}
                    t=float((np.datetime64(row['expiry'])-np.datetime64(day)).astype(float)/365)
                    try:
                        if state['model'] is None:raise ValueError('no timely qualified curve')
                        ratio=hedge_delta(state['model'],t,float(row['strike'])/float(row['spot']),kind)
                        if not np.isfinite(ratio):raise ValueError('nonfinite curve Delta')
                    except (ValueError,FloatingPointError) as error:
                        ratio=float(ndtr(np.log(float(row['spot'])/float(row['strike']))/(vol*np.sqrt(t))+.5*vol*np.sqrt(t)))
                        meta.update(fallback_source='RV20_BS',fallback_reason=str(error))
                actions[family,kind,ident]=float(ratio);metadata[family,kind,ident]=meta
    base_fields=['end','model','delta','cost_bp','execution','episode']
    book_fields=base_fields+['date','shares','traded','cost','accrued','paid','receivable','liability','cash','wealth',
                            'raw_decision','executed_ratio','pending_absolute_shares','fallback_source','source_date','stale_days']
    ratio_fields=base_fields+['date','ratio','raw_decision','pending_absolute_shares']
    stats=[];tmp_book=root/'daily_ledger.csv.tmp';tmp_ratios=root/'ratios.csv.tmp'
    with tmp_book.open('w',encoding='utf8',newline='') as book_file,tmp_ratios.open('w',encoding='utf8',newline='') as ratio_file:
        books=csv.DictWriter(book_file,fieldnames=book_fields);books.writeheader()
        ratios_out=csv.DictWriter(ratio_file,fieldnames=ratio_fields);ratios_out.writeheader()
        for family in ('bs','svi','ssvi','essvi'):
            for kind in (('black',) if family=='bs' else ('black','smile')):
                for execution,bp in [('same_close',0),('same_close',5),('same_close',10),('next_close',5)]:
                    for path in paths:
                        raw=[actions[family,kind,key(row)] for row in path[:-1]]
                        result=execute_path(path,raw,bp,execution)
                        ident=dict(end=int(end),model=family,delta=kind,cost_bp=bp,execution=execution,episode=int(path[0]['episode']))
                        used=[metadata[family,kind,key(row)] for row in path[:-1]]
                        fallbacks=sum(x['fallback_source'] not in ('same_day','current_quote') for x in used)
                        stats.append(dict(**ident,**result['diagnostics'],fallback_decisions=fallbacks))
                        for i,entry in enumerate(result['book']):
                            meta={k:used[i][k] if i<len(used) else None for k in ('fallback_source','source_date','stale_days')}
                            books.writerow(dict(**ident,**entry,**meta))
                            if i<len(raw):ratios_out.writerow(dict(**ident,date=entry['date'],ratio=entry['executed_ratio'],
                                raw_decision=raw[i],pending_absolute_shares=entry['pending_absolute_shares']))
    tmp_book.replace(root/'daily_ledger.csv');tmp_ratios.replace(root/'ratios.csv')
    frame=pd.DataFrame(stats);temporary=root/'path_pnl.csv.tmp';frame.to_csv(temporary,index=False);temporary.replace(root/'path_pnl.csv')
    from direct_dynamic import _groups
    grouped=_groups(frame.rename(columns={'delta':'hedge','cost_bp':'bp'})).rename(columns={'hedge':'delta','bp':'cost_bp'})
    temporary=root/'group_metrics.csv.tmp';grouped.to_csv(temporary,index=False);temporary.replace(root/'group_metrics.csv')
    ids=[int(path[0]['episode']) for path in paths]
    receipt=dict(status='static_delta_complete',complete=True,scope='static_direct_benchmarks_only',paths=len(paths),
        scenario_groups=28,episode_source_sha256=digest(source),
        path_ids_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),parents=parents,
        sources={name:digest(HERE/name) for name in ['direct_hedges.py','direct_dynamic.py','volatility_surfaces.py']},
        output_sha256={name:digest(root/name) for name in ['path_pnl.csv','daily_ledger.csv','ratios.csv','group_metrics.csv']},
        cached_frozen_actions=len(actions),stale_rule='all calendar days, five trading-day prefix, never future curves',
        fallback_rule='current RV20 of log((S+cash_dividend)/S_previous); min sigma .01')
    write_json(root/'status.json',receipt);return receipt


def run(end):
    root=OUT/f'direct/{end}';root.mkdir(parents=True,exist_ok=True);parents={}
    write_json(root/'status.json',dict(complete=False,status='running'))
    try:return _run(end,root,parents)
    except Exception as error:
        write_json(root/'status.json',dict(complete=False,status='failed',error_type=type(error).__name__,
                   error=str(error),parents=parents,source=digest(HERE/'direct_hedges.py')))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--end',type=int,required=True);run(p.parse_args().end)
