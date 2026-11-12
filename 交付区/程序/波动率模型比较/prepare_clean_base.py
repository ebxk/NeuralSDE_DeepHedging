"""Same cleaned quotes for native and original curves; isolated legacy encoder."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from study_core import OUT,PROJECT,HERE,CONFIG,SHARED_SURFACES,base_factors,bootstrap_legacy,digest,write_json


def fixed_constraints(path):
    with np.load(path,allow_pickle=False) as data:return {key:data[key] for key in ('A','b','grid_t','grid_k')}


def frozen_weights(path):
    with np.load(path,allow_pickle=False) as data:return data['weights'].copy()


def rebuild_day(quotes,fixed):
    from factor_quote_audit import fit_original_view
    result=fit_original_view(quotes,fixed['grid_t'],fixed['grid_k'],fixed['A'],fixed['b'],'full')
    values=quotes.spot.to_numpy(float)
    if not np.isfinite(values).all() or np.any(values<=0) or not np.allclose(values,values[0],rtol=0,atol=1e-12):
        raise ValueError('inconsistent observed spot within cleaned quote day')
    tenors=quotes.ttm.nunique();nks=len(np.unique(fixed['grid_k']))
    return dict(raw=np.array(result['raw_grid']),calls=np.array(result['grid']),spot=float(values[0]),
        repair_mean_abs=result['repair']['repair_mean_abs'],repair_l1=result['repair']['repair_l1'],repaired=result['repair']['repaired'],
        quote_outside_grid_fraction=result['out_of_grid_fraction'],
        strike_extrapolated_fraction=result['construction_strike_extrapolated_nodes']/(tenors*nks),
        tenor_extrapolated_fraction=result['construction_tenor_extrapolated_nodes']/len(fixed['grid_t']))


def _atomic_npz(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.npz.tmp')
    with temporary.open('wb') as handle:np.savez_compressed(handle,**arrays)
    temporary.replace(path)


def run_surfaces():
    started=time.perf_counter();status=OUT/'data/clean_base/status.json';parents={}
    write_json(status,dict(complete=False,status='running'))
    def parent(path):path=Path(path);parents[str(path)]=digest(path);return path
    try:
        from factor_quote_audit import clean_quotes
        quote_path=parent(OUT/'data/quotes.csv');q=clean_quotes(pd.read_csv(quote_path))
        constraints=parent(PROJECT/'工作区/数据/处理后/创新组合实验/surfaces.npz');fixed=fixed_constraints(constraints)
        if len(fixed['grid_t'])!=20 or not np.allclose(np.unique(fixed['grid_t']),np.array([14,30,60,90])/365,rtol=0,atol=1e-12):
            raise ValueError('original fixed 20-node time grid changed')
        parents[str(HERE/'factor_quote_audit.py')]=digest(HERE/'factor_quote_audit.py')
        parents[str(PROJECT/'工作区/当前研究/程序/扩展组合实验/extended_curve.py')]=digest(PROJECT/'工作区/当前研究/程序/扩展组合实验/extended_curve.py')
        dates=[];rows=[]
        for day,quotes in q.groupby('date',sort=True):
            dates.append(day);rows.append(rebuild_day(quotes,fixed))
        if len(dates)!=2646:raise ValueError('shared clean baseline requires all 2646 quote dates')
        arrays={key:np.array([row[key] for row in rows]) for key in rows[0]};arrays['dates']=np.array(dates)
        arrays.update(fixed)
        slack=float(np.min(arrays['calls']@fixed['A'].T-fixed['b']))
        if slack<-1e-7 or not np.isfinite(arrays['calls']).all():raise ValueError('shared clean baseline financial infeasibility')
        for name,sha in parents.items():
            if digest(name)!=sha:raise ValueError('baseline input changed during reconstruction: '+name)
        _atomic_npz(SHARED_SURFACES,**arrays)
        record=dict(complete=True,status='complete',source=digest(HERE/'prepare_clean_base.py'),parents=parents,
            quote_sha256=digest(quote_path),shared_surfaces_sha256=digest(SHARED_SURFACES),
            output_sha256={str(SHARED_SURFACES):digest(SHARED_SURFACES)},dates=len(dates),quotes=len(q),
            repaired_days=int(arrays['repaired'].sum()),repair_mean_abs=float(arrays['repair_mean_abs'].mean()),
            minimum_slack=slack,seconds=time.perf_counter()-started,
            rule='same already cleaned quotes as native SVI; original interpolation and fixed A,b L1 repair',
            old_fields_read=['constraints A,b,grid_t,grid_k only; no old calls, dates or spot'])
        write_json(status,record);print(json.dumps(record,ensure_ascii=False),flush=True);return record
    except Exception as error:
        write_json(status,dict(complete=False,status='failed',source=digest(HERE/'prepare_clean_base.py'),parents=parents,
                   error_type=type(error).__name__,error=str(error),seconds=time.perf_counter()-started));raise


def encode_frozen_basis(data,weights,train_end,decode):
    train=np.asarray(data['dates']).astype(str)<str(train_end+1)
    if not train.any():raise ValueError('no legal encoder training dates')
    if not np.isfinite(weights).all() or np.any(np.asarray(weights)<=0):raise ValueError('invalid frozen C1 coverage weights')
    shrunk=False
    try:fitted=decode(data,train,np.asarray(weights).copy())
    except (AssertionError,ValueError):
        fitted=decode(data,train,.5+.5*np.asarray(weights));shrunk=True
    if not all(np.isfinite(fitted[k]).all() for k in ('center','basis','xi','W','bounds')):raise ValueError('nonfinite original encoder')
    factor_slack=float(np.min(fitted['xi']@fitted['W'].T-fitted['bounds']))
    curves=fitted['center']+fitted['xi']@fitted['basis']
    financial=float(np.min(curves@data['A'].T-data['b']))
    if min(factor_slack,financial)<-1e-7:raise ValueError('clean encoder violates frozen financial domain')
    return fitted,shrunk


def encoder_worker(end):
    started=time.perf_counter();path=base_factors(end);status=path.with_suffix('.json');parents={}
    write_json(status,dict(complete=False,status='running'))
    def parent(p):p=Path(p);parents[str(p)]=digest(p);return p
    try:
        bootstrap_legacy()
        import representation
        import cvxopt
        cvxopt.solvers.options['show_progress']=False
        if 'torch' in sys.modules:raise ValueError('encoder must run in isolated legacy market interpreter, without PyTorch')
        shared_status=json.loads(parent(OUT/'data/clean_base/status.json').read_text(encoding='utf8'))
        if shared_status.get('complete') is not True or digest(SHARED_SURFACES)!=shared_status['shared_surfaces_sha256']:
            raise ValueError('shared clean baseline not complete/current')
        data=dict(np.load(parent(SHARED_SURFACES),allow_pickle=False))
        old=parent(PROJECT/f'工作区/数据/处理后/创新组合实验/{end}/C1.npz');weights=frozen_weights(old)
        train_end=next(fold[0] for fold in CONFIG['folds'] if fold[-1]==end)
        parent(representation.__file__)
        for p in sorted((PROJECT/'工作区/运行支持/临时文件/neuralSDE-marketmodel/marketmodel').glob('*.py')):parent(p)
        fitted,shrunk=encode_frozen_basis(data,weights,train_end,representation.decode)
        if fitted['basis'].shape!=(2,20):raise ValueError('clean original encoder must have two 20-node factors')
        for name,sha in parents.items():
            if digest(name)!=sha:raise ValueError('encoder input changed during fit: '+name)
        _atomic_npz(path,dates=data['dates'],spot=data['spot'],grid_t=data['grid_t'],grid_k=data['grid_k'],
                    shrunk=shrunk,source=np.full(len(data['dates']),'shared_clean_quotes'),**fitted)
        record=dict(complete=True,status='complete',source=digest(HERE/'prepare_clean_base.py'),parents=parents,
            shared_surfaces_sha256=digest(SHARED_SURFACES),train_end=train_end,training_before=str(train_end+1),
            original_weights_sha256=digest(old),shrunk=shrunk,dates=len(data['dates']),
            minimum_slack=float(np.min(fitted['xi']@fitted['W'].T-fitted['bounds'])),
            output_sha256={path.name:digest(path)},seconds=time.perf_counter()-started,
            old_fields_read=['C1 weights only; no old basis, xi, market weights or simulations'],torch_loaded=False)
        write_json(status,record);print(json.dumps(record,ensure_ascii=False),flush=True);return record
    except Exception as error:
        write_json(status,dict(complete=False,status='failed',source=digest(HERE/'prepare_clean_base.py'),parents=parents,
                   error_type=type(error).__name__,error=str(error),seconds=time.perf_counter()-started));raise


def run_encoder(end):
    interpreter=PROJECT/'工作区/运行支持/环境/market/python.exe'
    environment=dict(os.environ)
    for key in ('PYTHONHOME','PYTHONPATH','VIRTUAL_ENV'):environment.pop(key,None)
    subprocess.run([str(interpreter),'-u',str(HERE/'prepare_clean_base.py'),'--stage','encoder','--end',str(end),'--encoder-worker'],check=True,env=environment)
    return json.loads(base_factors(end).with_suffix('.json').read_text(encoding='utf8'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,choices=['surfaces','encoder'])
    parser.add_argument('--end',type=int);parser.add_argument('--encoder-worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.stage=='surfaces':run_surfaces()
    elif args.end is None:parser.error('--end is required for encoder')
    elif args.encoder_worker:encoder_worker(args.end)
    else:run_encoder(args.end)
