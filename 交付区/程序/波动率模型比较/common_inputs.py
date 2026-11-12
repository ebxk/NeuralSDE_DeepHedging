"""Common observed information and immutable matched simulator inputs."""
import numpy as np
import torch
from study_core import PROJECT,HERE,OUT,CONFIG,SHARED_SURFACES,base_factors,bootstrap_legacy,digest,write_json
from linear_market import shared_summary

def save_verified(path,arrays):
    if path.exists():
        with np.load(path,allow_pickle=False) as old:
            if set(old.files)!=set(arrays) or any(not np.array_equal(old[k],v,equal_nan=True) for k,v in arrays.items()):
                raise ValueError(f'Changed input arrays require archived generation: {path}')
        return
    tmp=path.with_suffix('.npz.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    tmp.replace(path)

def prepare_inputs(end):
    bootstrap_legacy()
    import pandas as pd
    from extended_data import Inputs
    from hedge_experiment import features
    old=PROJECT/f'工作区/数据/处理后/创新组合实验/{end}'
    import json
    cfg=json.loads((old/'config.json').read_text(encoding='utf-8-sig'))
    from artifact_guard import check_hashes
    fold=cfg['split_ends'];inp=Inputs(old,old/'C1',base_factors(end),fold)
    surf_file=SHARED_SURFACES
    surface=dict(np.load(surf_file,allow_pickle=False))
    summary=shared_summary(surface['calls'],np.unique(surface['grid_t']),np.unique(surface['grid_k']))
    lookup=dict(zip(surface['dates'],summary))
    root=OUT/f'inputs/{end}';root.mkdir(parents=True,exist_ok=True)
    training_features=[]
    for year in fold:
        paths,data=inp.observed(year)
        f=data['f'][...,:3].numpy();info=np.zeros((*f.shape[:2],4))
        for i,path in enumerate(paths):
            for j,row in enumerate(path[:-1]):info[i,j]=lookup[row['date']]
        merged=np.concatenate([f,info],axis=-1).astype('f')
        if year==fold[0]:training_features=merged[data['active'].numpy()]
        arrays={k:v.numpy() for k,v in data.items() if k not in ('f','tape')}
        arrays['f']=merged
        save_verified(root/f'real_{year}.npz',arrays)
    scale=np.std(training_features,axis=0).clip(1e-6);center=np.mean(training_features,axis=0)
    center[5:]=0;scale[5:]=1
    save_verified(root/'feature_scale.npz',dict(center=center,scale=scale))
    common=inp.common
    rep=inp.rep
    reference=OUT/f'markets/{end}/nsde/reference_common.npz'
    if not reference.exists():return
    fresh=dict(np.load(reference,allow_pickle=False))
    for name in ('time','warm','strike','premium'):
        if not np.array_equal(fresh[name].astype(common[name].dtype),common[name]):
            raise ValueError('Fresh reference changed shared contract origin data: '+name)
    for model in ('gbm',):
        s=fresh['gbm_new']
        sigma=float(fresh['gbm_sigma'])
        from volatility_surfaces import bs_call
        calls=bs_call(np.unique(rep['grid_t'])[:,None],np.unique(rep['grid_k'])[None,:],sigma)
        info=np.broadcast_to(shared_summary(calls,np.unique(rep['grid_t']),np.unique(rep['grid_k'])),(*s.shape,4))
        t=common['time'];tau=t[:,-1:]-t
        returns=np.diff(np.log(s),axis=1)
        rv=np.sqrt(252*np.mean(np.lib.stride_tricks.sliding_window_view(
            np.concatenate([common['warm'],returns],axis=1),20,axis=1)**2,axis=-1))
        base=features(s,common['strike'][:,None],tau,rv,np.empty((*s.shape,0)))[...,:3]
        f=np.concatenate([base[:,:-1],info[:,:-1]],axis=-1).astype('f')
        save_verified(root/f'sim_{model}.npz',dict(f=f,spot=s,unit=np.ones_like(s),
            active=np.diff(t,axis=1)>0,premium=common['premium'].astype('f'),
            strike_end=common['strike'].astype('f')))
    parents={str(p):digest(p) for p in [surf_file,base_factors(end),old/'episode_days.csv',old/'C1/common.npz',
        reference,OUT/f'markets/{end}/nsde/simulation.npz',OUT/f'markets/{end}/nsde/market.json']}
    import importlib
    sources={str(HERE/name):digest(HERE/name) for name in ('common_inputs.py','linear_market.py','study_core.py')}
    for name in ('extended_data','extended_curve','hedge_experiment','contract_ledger'):
        file=importlib.import_module(name).__file__;sources[str(file)]=digest(file)
    write_json(root/'identity.json',dict(parents=parents,fold=fold,
        shared_observation='same repaired raw quote grid for all generators',
        generator_sources='newly fitted reference market; frozen encoder and contract origins'))
    for model in ('gbm',):
        p=root/f'sim_{model}.npz'
        if p.exists():write_json(root/f'sim_{model}.ready.json',dict(complete=True,
            model=model,end=end,config=CONFIG,input_sha256=digest(p),parents=parents,
            sources=sources,
            identity_sha256=digest(root/'identity.json')))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--end',type=int,required=True)
    prepare_inputs(p.parse_args().end)
