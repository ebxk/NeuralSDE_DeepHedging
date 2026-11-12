"""Fresh nonlinear constrained joint NSDE on a frozen basis, never old weights."""
import argparse
import ast
import inspect
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from study_core import CONFIG,OUT,PROJECT,HERE,base_factors,bootstrap_legacy,digest,write_json
from linear_market import shared_summary
from verify_predecessor import assert_full_array_identity


def validate_original_head(source):
    widths=[int(node.args[0].value) for node in ast.walk(ast.parse(source))
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
            and node.func.attr=='Dense' and node.args and isinstance(node.args[0],ast.Constant)]
    if widths!=[64,64,7]:raise ValueError('fresh G1 requires original Dense64/Dense64/Dense7 nonlinear head')
    return widths


def load_shared_conditions(path):
    with np.load(path,allow_pickle=False) as data:
        return {key:data[key] for key in ('time','warm','strike','premium')}


def paired_shocks(n,h,seed=105):
    fine=np.random.default_rng(seed).normal(size=(n,h-1,4,3))
    return fine,fine.reshape(n,h-1,2,2,3).sum(3)/np.sqrt(2)


def check_factor_domain(rep,states=None,scale=None):
    states=np.asarray(rep['xi'] if states is None else states,float).reshape(-1,2)
    if not np.isfinite(states).all():raise ValueError('nonfinite factor states')
    weights=np.asarray(rep['W'])*np.asarray(np.ones(2) if scale is None else scale)
    minimum=np.inf
    for a in range(0,len(states),16384):
        minimum=min(minimum,float(np.min(states[a:a+16384]@weights.T-rep['bounds'])))
    if not np.isfinite(minimum) or minimum<-1e-7:raise ValueError('frozen factor financial domain violated: '+str(minimum))
    return float(minimum)


def closed_spot_parameters(rep,calendar,fold,events):
    dates=np.asarray(rep['dates']);dt=np.diff(dates.astype('datetime64[D]')).astype(float)/365
    indices=calendar.index.get_indexer(dates)
    legal=(dates[1:]<str(fold[0]+1))&(np.diff(indices)==1)&(indices[:-1]>=0)
    legal&=~np.isin(dates[1:],[event[0] for event in events])
    r=np.diff(np.log(rep['spot']))[legal];steps=dt[legal];n=int(.9*len(r))
    if n<2 or not np.isfinite(r).all() or np.any(steps<=0):raise ValueError('insufficient legal training increments')
    log_mu=float(r[:n].sum()/steps[:n].sum())
    sigma=float(np.sqrt(np.mean((r[:n]-log_mu*steps[:n])**2/steps[:n])))
    return dict(log_mu=log_mu,sigma=sigma,legal_training_transitions=len(r),fit_transitions=n,
                validation_transitions=len(r)-n,training_before=str(fold[0]+1),
                last_fit_date=str(dates[1:][legal][n-1]),nll_dates=dates[1:][legal][n:])


def _atomic_npz(path,**arrays):
    temporary=path.with_suffix('.npz.tmp')
    with temporary.open('wb') as handle:np.savez_compressed(handle,**arrays)
    temporary.replace(path)


def _moments(states,t):
    increments=np.diff(states,axis=1)[np.diff(t,axis=1)>0]
    return dict(quantiles=np.quantile(increments[:,0],[.01,.5,.99]).tolist(),covariance=np.cov(increments.T).tolist())


def run(end):
    start=time.perf_counter();directory=OUT/f'markets/{end}/nsde';directory.mkdir(parents=True,exist_ok=True)
    parents={};write_json(directory/'market.json',dict(complete=False,status='running',end=int(end)))
    def parent(path):
        path=Path(path);parents[str(path)]=digest(path);return path
    try:
        bootstrap_legacy();import market
        from contract_ledger import EVENTS
        market.tf.config.threading.set_intra_op_parallelism_threads(2)
        market.tf.config.threading.set_inter_op_parallelism_threads(2)
        original_source=inspect.getsource(market.fit_member);widths=validate_original_head(original_source)
        parent(market.__file__)
        base=PROJECT/f'工作区/数据/处理后/创新组合实验/{end}'
        factor_path=parent(base_factors(end));rep=dict(np.load(factor_path,allow_pickle=False))
        cfg=json.loads(parent(base/'config.json').read_text(encoding='utf-8-sig'))
        if cfg['split_ends']!=list(next(fold for fold in CONFIG['folds'] if fold[-1]==end)):
            raise ValueError('fresh reference fold configuration mismatch')
        if not np.all(np.diff(rep['dates'].astype('datetime64[D]'))>np.timedelta64(0,'D')):
            raise ValueError('frozen factor dates must be strictly increasing')
        domain=check_factor_domain(rep)
        spot_path=parent(PROJECT/'工作区/数据/原始数据/50ETF/510050_华夏上证50ETF.csv')
        calendar=pd.read_csv(spot_path,encoding='gb18030').set_index('交易日期')['收盘价(元)'].sort_index()
        episodes=pd.read_csv(parent(base/'episode_days.csv'))
        common=load_shared_conditions(parent(base/'C1/common.npz'))
        closed=closed_spot_parameters(rep,calendar,cfg['split_ends'],EVENTS)
        upstream=PROJECT/'工作区/运行支持/临时文件/neuralSDE-marketmodel/marketmodel'
        for path in sorted(upstream.glob('*.py')):parent(path)
        parent(PROJECT/'工作区/当前研究/程序/完整认购/contract_ledger.py')
        identity=dict(source=digest(HERE/'reference_market.py'),actual_original_market_source_sha256=digest(market.__file__),
            factor=digest(factor_path),episodes=digest(base/'episode_days.csv'),config=digest(base/'config.json'),
            rawspot=digest(spot_path),end=int(end),parents=parents,study_config=CONFIG,fit_config=cfg,
            basis_reused=True,market_weights_reused=False,market_simulations_reused=False,
            head_widths=widths,seed=2026,member=0,training_before=closed['training_before'],
            original_function='market.fit_member, unchanged two-hidden-layer nonlinear constrained joint likelihood')
        print(f'fresh G1 {end}: fitting original nonlinear Dense64/Dense64/Dense7, no old weight load',flush=True)
        fit_start=time.perf_counter()
        fit,nll,geometry=market.fit_member(rep,calendar,cfg,directory,2026,0)
        fit_seconds=time.perf_counter()-fit_start
        np.testing.assert_allclose([geometry[6],geometry[7]],[closed['log_mu'],closed['sigma']],rtol=0,atol=1e-14)
        t,warm,initial,strike,premium,returns=market.prepare_origins(episodes,rep['dates'],geometry[0],calendar,cfg)
        conditions={}
        for name,value in [('time',t),('warm',warm),('strike',strike),('premium',premium)]:
            conditions[name]=assert_full_array_identity(name,value,common[name])
        n,h=t.shape
        if n!=40960 or n!=CONFIG['simulation_train']+CONFIG['simulation_valid'] or cfg['origin_seed']!=104 or cfg['shock_seed']!=105:
            raise ValueError('fresh reference must preserve full budget and common seeds')
        fine,coarse=paired_shocks(n,h,105)
        print(f'fresh G1: fit done in {fit_seconds:.3f}s; simulating full {n}x{h} paths',flush=True)
        states,diag2=market.simulate(fit,initial,t,coarse,geometry,2,'reflect')
        full_slack=check_factor_domain(rep,states[:,:,1:],geometry[1])
        paired=slice(None,None,8)
        fine_states,diag4=market.simulate(fit,initial[paired],t[paired],fine[paired],geometry,4,'reflect')
        fine_slack=check_factor_domain(rep,fine_states[:,:,1:],geometry[1])
        numeric=dict(full2=diag2,paired4=diag4,moments2=_moments(states[paired],t[paired]),moments4=_moments(fine_states,t[paired]),
                     full_slack=full_slack,fine_slack=fine_slack,
                     paired_terminal_state_rmse=np.sqrt(np.mean((states[paired,-1]-fine_states[:,-1])**2,axis=0)).tolist())
        del fine,fine_states
        info=np.empty((n,h,4),dtype='f');ts=np.unique(rep['grid_t']);ks=np.unique(rep['grid_k'])
        for a in range(0,n,512):
            prices=rep['center']+(states[a:a+512,:,1:]*geometry[1])@rep['basis']
            info[a:a+512]=shared_summary(prices,ts,ks)
        spot=np.exp(states[:,:,0]).astype('f');latent=states[:,:,1:].astype('f')
        numeric['stored_float32_full_slack']=check_factor_domain(rep,latent,geometry[1])
        dt=np.diff(t,axis=1)
        gbm_log=np.c_[initial[:,0],initial[:,0,None]+np.cumsum(
            closed['log_mu']*dt+closed['sigma']*np.sqrt(dt)*coarse[:,:,:,0].sum(2)/np.sqrt(2),axis=1)]
        gbm_new=np.exp(gbm_log).astype('f');del coarse,states,gbm_log
        real_mu,real_vol=fit(geometry[0]);real_cov=real_vol@real_vol.transpose(0,2,1)
        if not np.isfinite(real_cov).all() or np.linalg.eigvalsh(real_cov).min()<-1e-10:raise ValueError('invalid fresh real covariance')
        _atomic_npz(directory/'simulation.npz',spot=spot,xi=latent,summary=info,time=common['time'],warm=common['warm'],strike=strike,premium=premium)
        _atomic_npz(directory/'reference_common.npz',time=common['time'],warm=common['warm'],strike=strike,premium=premium,
            gbm_new=gbm_new,real_dates=rep['dates'],real_xi=geometry[0],scale=geometry[1],real_cov=real_cov,real_mu=real_mu,
            gbm_sigma=np.asarray(closed['sigma']),gbm_log_mu=np.asarray(closed['log_mu']),
            return_dates=calendar.index.to_numpy(dtype=str),returns=returns,nll=nll,nll_dates=closed['nll_dates'],
            normalized_W=geometry[2],normalized_b=geometry[3],anchor=geometry[4],radius=geometry[5])
        for path,sha in parents.items():
            if digest(path)!=sha:raise ValueError('reference input/source changed during fit: '+path)
        closed_record={key:value for key,value in closed.items() if key!='nll_dates'}
        receipt=dict(complete=True,status='complete',identity=identity,rows=n,train=CONFIG['simulation_train'],valid=CONFIG['simulation_valid'],
            validation_nll=float(np.mean(nll)),nll_count=len(nll),fit_seconds=fit_seconds,seconds=time.perf_counter()-start,
            frozen_basis_minimum_slack=domain,diagnostics2=diag2,diagnostics4=diag4,numeric=numeric,
            common_conditions_exact=conditions,closed_gbm=closed_record,
            sources={name:digest(HERE/name) for name in ['reference_market.py','linear_market.py','study_core.py','verify_predecessor.py']},
            output_sha256={name:digest(directory/name) for name in ['simulation.npz','reference_common.npz','market_2026.weights.h5']},
            definition='new nonlinear constrained joint NSDE fit; frozen basis reused; no predecessor market result reuse')
        write_json(directory/'market.json',receipt);print(json.dumps({k:receipt[k] for k in ('complete','rows','validation_nll','seconds')}),flush=True)
        return receipt
    except Exception as error:
        write_json(directory/'market.json',dict(complete=False,status='failed',error_type=type(error).__name__,
                   error=str(error),parents=parents,source=digest(HERE/'reference_market.py'),seconds=time.perf_counter()-start))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--end',type=int,required=True)
    run(parser.parse_args().end)
