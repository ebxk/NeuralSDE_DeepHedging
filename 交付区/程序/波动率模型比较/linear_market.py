"""Low-capacity nested joint SDE and a single price-grid feature operator."""
import argparse
import inspect
import json
import sys
import numpy as np
from scipy.special import ndtr
from study_core import PROJECT,HERE,OUT,CONFIG,base_factors,bootstrap_legacy,digest,write_json

OLD_HEAD='''    model = tf.keras.Sequential([tf.keras.layers.Input((2,)),
        tf.keras.layers.Dense(64,activation="tanh"), tf.keras.layers.Dense(64,activation="tanh"),
        tf.keras.layers.Dense(7)])'''
NEW_HEAD='''    model = tf.keras.Sequential([tf.keras.layers.Input((2,)),
        tf.keras.layers.Dense(7)])'''

def coefficient_head_source(source):
    if source.count(OLD_HEAD)!=1:raise ValueError('Frozen joint SDE architecture changed')
    return source.replace(OLD_HEAD,NEW_HEAD)

def shared_summary(prices,ts,ks):
    """Same discrete normalized-price operator in simulation and observations."""
    prices=np.asarray(prices);ts=np.asarray(ts);ks=np.asarray(ks)
    prices=prices.reshape(*prices.shape[:-2],len(ts),len(ks)) if prices.shape[-2:]==(len(ts),len(ks)) else prices.reshape(*prices.shape[:-1],len(ts),len(ks))
    target=30/365;i=int(np.clip(np.searchsorted(ts,target)-1,0,len(ts)-2))
    weight=(target-ts[i])/(ts[i+1]-ts[i]);curve=prices[...,i,:]*(1-weight)+prices[...,i+1,:]*weight
    nodes=np.array([.95,1,1.05]);cols=[]
    for k in nodes:
        j=int(np.clip(np.searchsorted(ks,k)-1,0,len(ks)-2));u=(k-ks[j])/(ks[j+1]-ks[j])
        cols.append(curve[...,j]*(1-u)+curve[...,j+1]*u)
    p=np.stack(cols,axis=-1);lo=np.zeros_like(p);hi=np.full_like(p,10.)
    valid=np.isfinite(p)&(p>np.maximum(1-nodes,0)+1e-12)&(p<1)
    for _ in range(48):
        v=(lo+hi)/2;r=v*np.sqrt(target);d=-np.log(nodes)/r+r/2
        estimate=ndtr(d)-nodes*ndtr(d-r)
        lo=np.where(estimate<p,v,lo);hi=np.where(estimate>=p,v,hi)
    iv=(lo+hi)/2
    return np.stack([np.where(valid[...,1],iv[...,1],0),
        np.where(valid[...,0]&valid[...,2],iv[...,2]-iv[...,0],0),
        valid[...,1].astype(float),(valid[...,0]&valid[...,2]).astype(float)],axis=-1)

def fit_original(end,name='original_linear'):
    import time
    started=time.perf_counter()
    bootstrap_legacy()
    import market
    import pandas as pd
    market.tf.config.threading.set_intra_op_parallelism_threads(2)
    market.tf.config.threading.set_inter_op_parallelism_threads(2)
    old=PROJECT/f'工作区/数据/处理后/创新组合实验/{end}'
    out=OUT/f'markets/{end}/{name}';out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((old/'config.json').read_text(encoding='utf-8-sig'))
    factors=base_factors(end) if name=='original_linear' else OUT/f'representations/{end}/{name}.npz'
    rep=dict(np.load(factors,allow_pickle=False))
    spot=PROJECT/'工作区/数据/原始数据/50ETF/510050_华夏上证50ETF.csv'
    cal=pd.read_csv(spot,encoding='gb18030').set_index('交易日期')['收盘价(元)'].sort_index()
    source=coefficient_head_source(inspect.getsource(market.fit_member))
    scope=dict(vars(market));exec(compile(source,'<low capacity head>', 'exec'),scope)
    fit,nll,geom=scope['fit_member'](rep,cal,cfg,out,2026,0)
    episodes=pd.read_csv(old/'episode_days.csv')
    t,warm,start,strike,premium,returns=market.prepare_origins(episodes,rep['dates'],geom[0],cal,cfg)
    n,h=t.shape
    shocks4=np.random.default_rng(cfg['shock_seed']).normal(size=(n,h-1,4,3))
    shocks2=shocks4.reshape(n,h-1,2,2,3).sum(3)/np.sqrt(2)
    states,diag=market.simulate(fit,start,t,shocks2,geom,2,'reflect')
    fine,fdiag=market.simulate(fit,start[::8],t[::8],shocks4[::8],geom,4,'reflect')
    scale=geom[1]
    actual=states[...,1:]*scale
    prices=rep['center']+actual@rep['basis']
    info=shared_summary(prices,np.unique(rep['grid_t']),np.unique(rep['grid_k']))
    np.savez_compressed(out/'simulation.npz',spot=np.exp(states[...,0]).astype('f'),
        summary=info.astype('f'),xi=states[...,1:].astype('f'),time=t,warm=warm,
        strike=strike,premium=premium)
    slack=min(float((block@rep['W'].T-rep['bounds']).min()) for block in np.array_split(actual.reshape(-1,2),128))
    if slack < -1e-7 or not np.isfinite(states).all():raise ValueError('Invalid simulated factor grid')
    parents={str(p):digest(p) for p in (factors,old/'episode_days.csv',old/'config.json',spot)}
    write_json(out/'market.json',dict(status='complete',complete=True,validation_nll=float(nll.mean()),
        diagnostics2=diag,diagnostics4=fdiag,rows=n,train=32768,valid=8192,
        full_grid_minimum_slack=slack,parents=parents,
        seconds=time.perf_counter()-started,
        paired_terminal_state_rmse=np.sqrt(np.mean((states[::8,-1]-fine[:,-1])**2,axis=0)).tolist(),
        sources={str(p):digest(p) for p in (HERE/'linear_market.py',__import__('pathlib').Path(market.__file__))},
        output_sha256={p.name:digest(p) for p in (out/'simulation.npz',out/'market_2026.weights.h5')},
        identity=dict(source=digest(HERE/'linear_market.py'),factor=digest(factors),
            episodes=digest(old/'episode_days.csv'),config=digest(old/'config.json')),
        definition='no hidden coefficient head; original nonlinear output and boundary transforms'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--end',type=int,required=True)
    p.add_argument('--model',choices=['original_linear','svi_linear','ssvi_linear','essvi_linear'],default='original_linear')
    args=p.parse_args()
    try:fit_original(args.end,args.model)
    except BaseException as error:
        import traceback
        numeric=isinstance(error,(FloatingPointError,np.linalg.LinAlgError,AssertionError))
        factor=base_factors(args.end) if args.model=='original_linear' else OUT/f'representations/{args.end}/{args.model}.npz'
        write_json(OUT/f'markets/{args.end}/{args.model}/market.json',dict(complete=False,
            status='numerical_failed' if numeric else 'failed',source=digest(HERE/'linear_market.py'),
            end=args.end,model=args.model,config=CONFIG,
            parents={str(factor):digest(factor)} if factor.exists() else {},
            error_type=type(error).__name__,reason=traceback.format_exc()))
        raise
