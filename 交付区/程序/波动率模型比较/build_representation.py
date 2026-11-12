"""Training-only two-factor compression of the three calibrated curve families."""
import argparse
import json
import numpy as np
import pandas as pd
from study_core import OUT,PROJECT,SHARED_SURFACES,bootstrap_legacy,write_json,digest
from volatility_surfaces import price_surface,bs_call

def run(end,kind):
    bootstrap_legacy()
    from representation import decode
    from contract_ledger import EVENTS
    from arbitragerepair.repair import l1
    old=PROJECT/f'工作区/数据/处理后/创新组合实验/{end}'
    cfg=json.loads((old/'config.json').read_text(encoding='utf-8-sig'))
    original=dict(np.load(SHARED_SURFACES,allow_pickle=False))
    w=np.load(old/'C1.npz')['weights']
    calendar=pd.read_csv(PROJECT/'工作区/数据/原始数据/50ETF/510050_华夏上证50ETF.csv',
        encoding='gb18030').set_index('交易日期')['收盘价(元)'].sort_index()
    dividends=pd.Series({e[0]:float(e[3]) for e in EVENTS}).reindex(calendar.index,fill_value=0.)
    daily=np.log((calendar+dividends)/calendar.shift());rv=np.sqrt(252*daily.pow(2).rolling(20).mean())
    calls=[];sources=[];last=None;stale=0;fallback=0;parents={}
    for day in original['dates']:
        receipt=OUT/f'surfaces/{kind}/{day}/full.json'
        if not receipt.exists():raise FileNotFoundError(f'Calibration still pending: {receipt}')
        r=json.loads(receipt.read_text(encoding='utf8'))
        parents[str(receipt)]=digest(receipt)
        if r['status']=='complete':last=r['model'];stale=0
        else:stale+=1
        if last is not None and stale<=5:
            curve=price_surface(last,original['grid_t'],original['grid_k']);source='current' if stale==0 else 'past_state'
        else:
            curve=bs_call(original['grid_t'],original['grid_k'],max(.01,float(rv.loc[day])));source='rv_fallback';fallback+=1
        margin=original['A']@curve-original['b']
        repaired=curve+l1(original['A'],original['b'],curve) if margin.min() < -1e-9 else curve
        if (original['A']@repaired-original['b']).min() < -1e-7:raise ValueError('Infeasible repaired price grid')
        calls.append(repaired);sources.append(source)
    original['calls']=np.array(calls)
    train=original['dates']<str(cfg['split_ends'][0]+1)
    shrunk=False
    try:fitted=decode(original,train,w)
    except (AssertionError,ValueError):
        fitted=decode(original,train,.5+.5*w);shrunk=True
    root=OUT/f'representations/{end}';root.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(root/f'{kind}_linear.npz',dates=original['dates'],spot=original['spot'],
        grid_t=original['grid_t'],grid_k=original['grid_k'],source=np.array(sources),shrunk=shrunk,**fitted)
    write_json(root/f'{kind}_linear.json',dict(complete=True,train_end=cfg['split_ends'][0],
        source=digest(__file__),parents={**parents,
            str(old/'C1.npz'):digest(old/'C1.npz'),str(old/'config.json'):digest(old/'config.json'),
            str(SHARED_SURFACES):digest(SHARED_SURFACES)},
        output_sha256=digest(root/f'{kind}_linear.npz'),
        fallback_days=fallback,shrunk=shrunk,original_weights_sha256=digest(old/'C1.npz'),
        finite_grid_constraints=True,continuous_dynamic_no_arbitrage_proven=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--end',type=int,required=True)
    p.add_argument('--kind',choices=['svi','ssvi','essvi'],required=True);args=p.parse_args();run(args.end,args.kind)
