"""Read-only source validation, historical terms and shared quote split."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from study_core import PROJECT,OUT,CONFIG,registry,digest,frozen_json,write_json,bootstrap_legacy,quote_mask

def prepare():
    bootstrap_legacy()
    from contract_ledger import terms
    raw=PROJECT/'工作区/数据/原始数据/50ETF'
    op=raw/'510050.XSHG_option_full.csv';sp=raw/'510050_华夏上证50ETF.csv'
    expected=json.loads((raw/'数据清单.json').read_text(encoding='utf-8-sig'))
    # Compare copied bytes to the original source receipts, independent of inventory schema.
    hashes={op.name:digest(op),sp.name:digest(sp)}
    required={op.name:'fdc74c64318368f911c0e8fae02f9155873d559b9ce94f1aa0e6a3e6d615ef9f',
        sp.name:'ab812f30a1a4036e6d6f8dd3e650fcce81b4e46470a72b5aca24f0d4fc74bef3'}
    if hashes!=required: raise ValueError('Raw input identity mismatch')
    frozen_json(OUT/'registry.json',dict(config=CONFIG,jobs=registry(),raw_sha256=hashes))
    cols=['date','code','trading_code','list_date','expire_date','contract_type','close','volume','position']
    q=pd.read_csv(op,usecols=cols)
    spot=pd.read_csv(sp,encoding='gb18030').set_index('交易日期')['收盘价(元)']
    q=q[q.contract_type.eq('CO')].copy()
    q['spot']=q.date.map(spot)
    effective=[terms(r,r['date']) for r in q.to_dict('records')]
    q['strike']=[float(x[0]) for x in effective];q['unit']=[x[1] for x in effective]
    q['ttm']=(pd.to_datetime(q.expire_date)-pd.to_datetime(q.date)).dt.days/365
    q['k']=q.strike/q.spot;q['price']=q.close/q.spot
    good=(q.close.gt(0)&q.volume.gt(0)&q.position.gt(0)&q.ttm.gt(0)&q.spot.gt(0))
    bounds=(q.price>=np.maximum(1-q.k,0)-1e-10)&(q.price<=1+1e-10)
    reason=np.where(~good,'liquidity_or_expired',np.where(~bounds,'zero_carry_price_bound',''))
    exclusions=q.loc[reason!='',['date','code']].copy();exclusions['reason']=reason[reason!='']
    folder=OUT/'data';folder.mkdir(parents=True,exist_ok=True)
    exclusions.to_csv(folder/'quote_exclusions.csv',index=False)
    q=q[good&bounds].sort_values(['date','expire_date','strike','volume','code'],
        ascending=[True,True,True,False,True]).drop_duplicates(['date','expire_date','strike'])
    q['holdout']=quote_mask(q.date,q.expire_date,q.strike)
    q.to_csv(folder/'quotes.csv',index=False)
    sample_counts={}
    for train,valid,end in CONFIG['folds']:
        old=PROJECT/f'工作区/数据/处理后/创新组合实验/{end}/episode_days.csv'
        episodes=pd.read_csv(old)
        counts=episodes.groupby('split_end').episode.nunique().to_dict()
        if end==2025 and [counts[y] for y in (train,valid,end)]!=[2626,387,773]:
            raise ValueError(f'Sample count mismatch: {counts}')
        sample_counts[str(end)]={str(k):int(v) for k,v in counts.items()}
        frozen_json(folder/f'episodes_{end}.json',dict(sha256=digest(old),counts=sample_counts[str(end)]))
    write_json(folder/'preparation.json',dict(complete=True,raw_sha256=hashes,
        source=digest(Path(__file__)),quotes_sha256=digest(folder/'quotes.csv'),
        accepted_quotes=len(q),excluded_quotes=len(exclusions),dates=q.date.nunique(),
        holdout_quotes=int(q.holdout.sum()),sample_counts=sample_counts,
        modeling='F=S; zero financing and continuous yield; actual dividends in real ledger'))
    return q

if __name__=='__main__': print(prepare().shape,flush=True)
