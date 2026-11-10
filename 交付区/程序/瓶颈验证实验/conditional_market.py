"""Frozen joint NSDE inference, paired distribution diagnostics and finite calibration.

No market weights are fitted or overwritten. All exclusions apply only to the
conditional diagnostic, never to hedge evaluation paths.
"""
import argparse
import contextlib
import inspect
import itertools
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.spatial.distance import pdist
from scipy.special import ndtr


def calibrated(coefficients, amplitudes):
    amplitudes=np.asarray(amplitudes,dtype=float)
    if amplitudes.shape!=(3,) or np.any(amplitudes<=0):
        raise ValueError('Three positive Brownian-direction amplitudes required')
    def evaluate(x):
        mu,sigma=coefficients(x)
        new_sigma=sigma*amplitudes[None,None,:]
        mu=mu.copy()
        mu[:,0]+=.5*((sigma[:,0]**2).sum(1)-(new_sigma[:,0]**2).sum(1))
        return mu,new_sigma
    return evaluate


def energy_score(samples, observed):
    samples=np.asarray(samples)
    return float(np.linalg.norm(samples-observed,axis=-1).mean()
                 -.5*pdist(samples).mean())


def crps(samples, observed):
    samples=np.asarray(samples)
    return float(np.abs(samples-observed).mean()-.5*pdist(samples[:,None],metric='cityblock').mean())


def numerical_check(coefficients,geometry,amplitudes):
    from scipy.spatial import HalfspaceIntersection
    _,_,w,b,anchor,radius,_,_=geometry
    vertices=HalfspaceIntersection(np.c_[-w,b],anchor).intersections
    points=[];faces=[]
    for i in range(len(w)):
        face=vertices[np.abs(vertices@w[i]-b[i])<1e-7]
        if len(face):points.append(face.mean(0));faces.append(i)
    points=np.array(points)
    old_mu,old_vol=coefficients(points)
    mu,vol=calibrated(coefficients,amplitudes)(points)
    covariance=vol@vol.transpose(0,2,1)
    tangent=np.einsum('bi,bij->bj',w[faces],vol[:,1:])
    old_tangent=np.einsum('bi,bij->bj',w[faces],old_vol[:,1:])
    inward=np.einsum('bi,bi->b',w[faces],mu[:,1:])
    eig=np.linalg.eigvalsh(covariance)
    if not (np.isfinite(mu).all() and np.isfinite(vol).all() and eig.min()>=-1e-9):
        raise AssertionError('Nonfinite or non-PSD calibrated boundary covariance')
    if np.max(np.abs(tangent))>max(1e-6,1.251*np.max(np.abs(old_tangent))):
        raise AssertionError('Calibration damaged original boundary tangency')
    if inward.min() < -1e-7:
        raise AssertionError('Original boundary drift is not inward')
    return dict(boundary_points=len(points),minimum_covariance_eigenvalue=float(eig.min()),
        maximum_tangent_diffusion=float(np.max(abs(tangent))),minimum_inward_drift=float(inward.min()),
        preserves_arithmetic_spot_drift=bool(np.allclose(mu[:,0]+.5*covariance[:,0,0],
            old_mu[:,0]+.5*(old_vol[:,0]**2).sum(1),atol=1e-12)))


def bootstrap(project):
    program=project/'工作区/当前研究/程序'
    sys.path[:0]=[str(program/'创新组合实验'),str(program/'完整认购'),
                  str(project/'工作区/运行支持/临时文件/neuralSDE-marketmodel')]
    return program


def load_frozen(project, end):
    """Reconstruct the original exact preprocessing and load saved TF weights.

    The original function is reused with its entire fit/save section replaced
    by load_weights. The replacement is guarded so upstream source drift fails.
    """
    import pandas as pd
    bootstrap(project)
    import market
    try:
        market.tf.config.threading.set_intra_op_parallelism_threads(2)
        market.tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass  # A previous frozen member already initialized the same process.
    base=project/'工作区/数据/处理后/创新组合实验'/str(end)
    rep=dict(np.load(base/'C1.npz',allow_pickle=False))
    common=dict(np.load(base/'C1/common.npz',allow_pickle=False))
    cfg=json.loads((base/'config.json').read_text(encoding='utf-8-sig'))
    spot=Path('D:/life/study/quant/投资/数据/ETF/510050_华夏上证50ETF.csv')
    cal=pd.read_csv(spot,encoding='gb18030').set_index('交易日期')['收盘价(元)'].sort_index()
    code=inspect.getsource(market.fit_member)
    start=code.index('    history = model.fit(')
    stop=code.index('    # NLL is evaluated directly',start)
    code=code[:start]+"    model.load_weights(str(out / f'market_{seed}.weights.h5'))\n"+code[stop:]
    if '.fit(' in code or 'save_weights(' in code:
        raise AssertionError('Inference loader contains a fitting or saving call')
    scope=dict(vars(market))
    exec(compile(code,'<frozen original market loader>','exec'),scope)
    fit,nll,geometry=scope['fit_member'](rep,cal,cfg,base/'C1',2026,0)
    mu,vol=fit(common['real_xi'])
    np.testing.assert_allclose(geometry[0],common['real_xi'],atol=1e-12,rtol=0)
    np.testing.assert_allclose(vol@vol.transpose(0,2,1),common['real_cov'],atol=1e-10,rtol=1e-8)
    from prepare_common import digest
    upstream=project/'工作区/运行支持/临时文件/neuralSDE-marketmodel'
    identity=dict(tensorflow=market.tf.__version__,numpy=np.__version__,coefficients_covariance_max_error=float(np.max(np.abs(
        vol@vol.transpose(0,2,1)-common['real_cov']))),files={str(p):digest(p) for p in
        [base/'C1.npz',base/'C1/common.npz',base/'C1/market_2026.weights.h5',
         base/'episode_days.csv',base/'config.json',Path(market.__file__),Path(__file__),
         project/'工作区/当前研究/程序/扩展组合实验/extended_curve.py',
         *sorted((upstream/'marketmodel').glob('*.py'))]})
    return fit,geometry,rep,common,cfg,cal,base,identity


def verify_original_paths(base,fit,geometry,common,cfg,count=16):
    import pandas as pd
    from market import simulate
    episodes=pd.read_csv(base/'episode_days.csv')
    origins=[g for _,g in episodes.groupby('episode') if g.iloc[0].split_end==cfg['split_ends'][0]]
    selected=np.random.default_rng(cfg['origin_seed']).integers(len(origins),size=len(common['time']))[:count]
    lookup={d:i for i,d in enumerate(common['real_dates'])}
    h=common['time'].shape[1];t=np.empty((count,h));start=np.empty((count,3))
    for i,j in enumerate(selected):
        p=origins[j];dates=p.date.to_numpy(dtype='datetime64[D]');tt=(dates-dates[0]).astype(float)/365
        t[i,:len(tt)]=tt;t[i,len(tt):]=tt[-1]
        start[i]=[np.log(p.iloc[0].spot),*common['real_xi'][lookup[p.iloc[0].date]]]
    fine=np.random.default_rng(cfg['shock_seed']).normal(size=(count,h-1,4,3))
    noise=fine.reshape(count,h-1,2,2,3).sum(3)/np.sqrt(2)
    local_h=max(len(origins[j]) for j in selected)
    states,diagnostics=simulate(fit,start,t[:,:local_h],noise[:,:local_h-1],geometry,2,'reflect')
    with np.load(base/'C1/single.npz') as old:
        actual_spot=np.exp(states[:,:,0]).astype('f');actual_xi=states[:,:,1:].astype('f')
        np.testing.assert_allclose(actual_spot,old['spot'][:count,:local_h],rtol=2e-6,atol=1e-7)
        # Small TF float32 matrix kernels accumulate sub-micro errors through
        # repeated boundary reflections; this is a numerical identity check,
        # never a relaxation of a financial inequality.
        np.testing.assert_allclose(actual_xi,old['xi'][:count,:local_h],rtol=0,atol=1e-5)
        error_spot=float(np.max(np.abs(actual_spot-old['spot'][:count,:local_h])))
        error_xi=float(np.max(np.abs(actual_xi-old['xi'][:count,:local_h])))
    return dict(paths=count,spot_max_error=error_spot,factor_max_error=error_xi,
        exact_float32_path_identity=error_spot==0 and error_xi==0,spot_rtol=2e-6,spot_atol=1e-7,
        factor_rtol=0,factor_atol=1e-5,diagnostics=diagnostics)


def curve_price(spot,strike,tau,xi,rep,scale,project):
    """Use precisely the existing curve function without importing PyTorch."""
    source=project/'工作区/当前研究/程序/扩展组合实验/extended_curve.py'
    code=source.read_text(encoding='utf-8-sig')
    scope={'np':np}
    exec(code[code.index('def curve('):code.index('\ndef bs_delta(')],scope)
    return scope['curve'](np.asarray(tau),np.asarray(strike)/spot,xi,rep,scale)[0]*spot


def delta(spot,strike,quote,tau):
    if not (tau>0 and max(spot-strike,0)<quote<spot):
        return None
    lo,hi=1e-8,12.
    for _ in range(40):
        v=(lo+hi)/2
        d=np.log(spot/strike)/v+v/2
        price=spot*ndtr(d)-strike*ndtr(d-v)
        if price<quote:lo=v
        else:hi=v
    return float(ndtr(np.log(spot/strike)/((lo+hi)/2)+(lo+hi)/4))


def windows(base,rep,common,cfg,cal,split,limit=128):
    """One contract per origin date; all 20 labels stay within the same partition."""
    import pandas as pd
    from contract_ledger import EVENTS
    df=pd.read_csv(base/'episode_days.csv',dtype={'date':str})
    target=cfg['split_ends'][0 if split=='train' else 1]
    df=df[df.split_end==target]
    dates=common['real_dates']; positions={v:i for i,v in enumerate(dates)}
    calendar={v:i for i,v in enumerate(cal.index)}
    legal=np.array([i for i in range(len(dates)-1)
        if dates[i+1]<str(cfg['split_ends'][0]+1) and
        calendar[dates[i+1]]-calendar[dates[i]]==1 and dates[i+1] not in [e[0] for e in EVENTS]])
    cutoff=legal[int(.9*len(legal))-1]+1
    exdays={e[0] for e in EVENTS}
    found={}; reasons={}
    def reject(reason):reasons[reason]=reasons.get(reason,0)+1
    for _,p in df.groupby('episode',sort=True):
        p=p.reset_index(drop=True)
        for k in range(len(p)-20):
            first,last=p.iloc[k],p.iloc[k+20]
            if first.date in found:continue
            i=positions.get(first.date,-1)
            if i<0 or i+20>=len(dates):reject('missing_factor');continue
            if split=='train' and i+20>=cutoff:reject('training_calibration_boundary');continue
            expected=dates[i:i+21]
            if not np.array_equal(p.date.iloc[k:k+21].to_numpy(),expected):reject('factor_gap');continue
            if any(calendar[expected[j+1]]-calendar[expected[j]]!=1 for j in range(20)):
                reject('calendar_gap');continue
            if any(d in exdays for d in expected[1:]):reject('ex_dividend');continue
            part=p.iloc[k:k+21]
            if part.unit.nunique()!=1 or part.strike.nunique()!=1:reject('term_adjustment');continue
            tau=(np.datetime64(first.expiry)-np.datetime64(first.date)).astype(float)/365
            dd=delta(float(first.spot),float(first.strike),float(first.option_close),tau)
            if dd is None or (part.option_close<=0).any():reject('invalid_quote');continue
            found[first.date]=dict(date=first.date,index=i,spot=float(first.spot),strike=float(first.strike),
                tau=float(tau),delta=dd,quotes=part.option_close.to_numpy(float),
                observed_spot=part.spot.to_numpy(float),episode=str(first.episode))
    selected=[found[d] for d in sorted(found)]
    if len(selected)>limit:
        selected=[selected[i] for i in np.linspace(0,len(selected)-1,limit,dtype=int)]
    return selected,dict(eligible_origin_dates=len(found),selected=len(selected),exclusions=reasons,
        diagnostic_scope='20-transition complete no-event windows; not all hedge states')


def evaluate_origins(origins,fit,geometry,rep,common,project,count,seed,substeps=2):
    from market import simulate
    if not origins:raise ValueError('No valid conditional origin')
    n=len(origins); starts=np.repeat(np.array([[np.log(o['spot']),*common['real_xi'][o['index']]] for o in origins]),count,axis=0)
    clocks=np.array([((common['real_dates'][o['index']:o['index']+21].astype('datetime64[D]')-
                      np.datetime64(o['date'])).astype(float)/365) for o in origins])
    clocks=np.repeat(clocks,count,axis=0)
    fine=np.random.default_rng(seed).normal(size=(n*count,20,4,3))
    shocks=fine if substeps==4 else fine.reshape(n*count,20,2,2,3).sum(3)/np.sqrt(2)
    with contextlib.redirect_stdout(sys.stderr):
        states,diagnostics=simulate(fit,starts,clocks,shocks,geometry,substeps,'reflect')
    # Scales use original training one-day transitions only, frozen across horizons.
    # geometry uses training-normalized factors; log-return scale is original fit sigma.
    scales=np.array([max(geometry[7]/np.sqrt(252),1e-5),1/np.sqrt(252),1/np.sqrt(252)])
    result=[]
    for oi,o in enumerate(origins):
        block=states[oi*count:(oi+1)*count]; ix=o['index']; labels=[]
        for h in (1,5,20):
            obs=np.r_[np.log(o['observed_spot'][h]/o['spot']),common['real_xi'][ix+h]-common['real_xi'][ix]]
            inc=block[:,h]-block[:,0]
            s=np.exp(block[:,h,0]); elapsed=clocks[oi*count,h]
            price0=curve_price(np.array([o['spot']]),o['strike'],np.array([o['tau']]),
                               common['real_xi'][ix:ix+1],rep,common['scale'],project)[0]
            price=curve_price(s,o['strike'],np.full(count,max(0,o['tau']-elapsed)),block[:,h,1:],rep,common['scale'],project)
            residual=(price-price0-o['delta']*(s-o['spot']))/o['spot']
            target=(o['quotes'][h]-o['quotes'][0]-o['delta']*(o['observed_spot'][h]-o['spot']))/o['spot']
            recfuture=curve_price(np.array([o['observed_spot'][h]]),o['strike'],np.array([max(0,o['tau']-elapsed)]),
                    common['real_xi'][ix+h:ix+h+1],rep,common['scale'],project)[0]
            rec=(recfuture-price0-o['delta']*(o['observed_spot'][h]-o['spot']))/o['spot']
            low,high=np.quantile(residual,[.025,.975])
            labels.append(dict(horizon=h,energy=energy_score(inc/(scales*np.sqrt(h)),obs/(scales*np.sqrt(h))),crps=crps(residual,target),
                quote_residual=float(target),reconstructed_residual=float(rec),reconstruction_increment=float(target-rec),
                covered=bool(low<=target<=high),predicted_interval=[float(low),float(high)],
                halves=[dict(energy=energy_score(inc[sl]/(scales*np.sqrt(h)),obs/(scales*np.sqrt(h))),
                    crps=crps(residual[sl],target),covered=bool(np.quantile(residual[sl],.025)<=target<=np.quantile(residual[sl],.975)))
                    for sl in (slice(0,count//2),slice(count//2,None))],
                simulated_covariance=np.cov(inc.T).tolist(),observed_increment=obs.tolist(),
                simulated_quantiles=np.quantile(inc,[.01,.5,.99],axis=0).tolist()))
        result.append(dict(date=o['date'],episode=o['episode'],scores=labels))
    summary={k:float(np.mean([x[k] for r in result for x in r['scores']])) for k in ('energy','crps','covered')}
    halves=[{k:float(np.mean([x['halves'][i][k] for r in result for x in r['scores']]))
             for k in ('energy','crps','covered')} for i in (0,1)]
    return dict(summary=summary,halves=halves,origins=result,boundary=diagnostics,count=count,seed=seed,substeps=substeps)


def run(args):
    from itertools import product
    project=args.project.resolve(); bootstrap(project)
    from prepare_common import digest,write_json
    output=args.out.resolve()/'markets'/str(args.end); output.mkdir(parents=True,exist_ok=True)
    fit,geom,rep,common,cfg,cal,base,identity=load_frozen(project,args.end)
    identity['paired_original_paths']=verify_original_paths(base,fit,geom,common,cfg)
    reference_mu,reference_vol=fit(common['real_xi'])
    np.savez(output/'reference_coefficients.npz',real_dates=common['real_dates'],mu=reference_mu,
             cov=reference_vol@reference_vol.transpose(0,2,1))
    shutil.copyfile(base/'C1/market.json',output/'reference_market.json')
    if args.reference_only:
        write_json(output/'reference_identity.json',identity)
        return
    train,train_info=windows(base,rep,common,cfg,cal,'train')
    valid,valid_info=windows(base,rep,common,cfg,cal,'validation')
    identity['config']=dict(amplitudes=[.8,1,1.25],origins=128,fit_scenarios=256,check_scenarios=1024,
        fit_seed=20261110,check_seed=20261111)
    marker=output/'calibration.json'
    if marker.exists() and json.loads(marker.read_text())['identity']==identity:
        previous=json.loads(marker.read_text())
        if args.export and previous['prediction_gate']:
            export(project,output,base,fit,geom,rep,common,cfg,previous)
        return
    records=[]
    for a in product((.8,1.,1.25),repeat=3):
        numeric=numerical_check(fit,geom,a)
        path=output/('fit_'+'_'.join(map(str,a))+'.json')
        result=json.loads(path.read_text()) if path.exists() else None
        if result is None or result.get('identity')!=identity:
            result=evaluate_origins(train,calibrated(fit,a),geom,rep,common,project,256,20261110)
            result['identity']=identity
            write_json(path,result)
        records.append(dict(amplitudes=list(a),numeric=numeric,**result['summary']))
        print('calibration fit',args.end,a,result['summary'],flush=True)
    reference=next(r for r in records if r['amplitudes']==[1,1,1])
    distance=lambda r:sum(np.log(r['amplitudes'])**2)
    stat=min(records,key=lambda r:(r['energy'],distance(r)))
    hedge=min(records,key=lambda r:(r['energy']/reference['energy']+r['crps']/reference['crps'],distance(r)))
    checks={}; half_checks={}; validation_records={}
    choices=[('identity',[1,1,1]),('stat',stat['amplitudes']),('hedge',hedge['amplitudes'])]
    for count in (1024,2048):
        for name,a in choices:
            path=output/f'validation_{name}_{count}.json'
            result=json.loads(path.read_text()) if path.exists() else None
            if result is None or result.get('identity')!=identity or result.get('amplitudes')!=a:
                result=evaluate_origins(valid,calibrated(fit,a),geom,rep,common,project,count,20261111)
                result.update(identity=identity,amplitudes=a)
                write_json(path,result)
            checks[name]=result['summary'];half_checks[name]=result['halves'];validation_records[name]=result
        stable={name:all((half_checks[name][0][key]-half_checks['identity'][0][key])*
                        (half_checks[name][1][key]-half_checks['identity'][1][key])>=0
                         for key in ('energy','crps')) for name in ('stat','hedge')}
        if all(stable.values()):break
    ref=checks['identity']; gates={}
    for name in ('stat','hedge'):
        s=checks[name];gates[name]=bool(stable[name] and (s['energy']<=.99*ref['energy'] or s['crps']<=.99*ref['crps']) and
            s['energy']<=1.01*ref['energy'] and s['crps']<=1.01*ref['crps'] and
            abs(s['covered']-.95)<=abs(ref['covered']-.95)+.01)
    fine=evaluate_origins(valid[:min(16,len(valid))],fit,geom,rep,common,project,1024,20261111,4)
    coarse=evaluate_origins(valid[:min(16,len(valid))],fit,geom,rep,common,project,1024,20261111,2)
    write_json(output/'paired_four_substeps.json',fine)
    write_json(output/'paired_two_substeps.json',coarse)
    moments=[]
    for j,horizon in enumerate((1,5,20)):
        c2=np.mean([o['scores'][j]['simulated_covariance'] for o in coarse['origins']],axis=0)
        c4=np.mean([o['scores'][j]['simulated_covariance'] for o in fine['origins']],axis=0)
        moments.append(dict(horizon=horizon,mean_conditional_covariance2=c2.tolist(),
            mean_conditional_covariance4=c4.tolist(),relative_frobenius_change=float(
                np.linalg.norm(c4-c2)/max(np.linalg.norm(c2),1e-15))))
    report=dict(identity=identity,train_windows=train_info,validation_windows=valid_info,
        candidates=records,stat=stat['amplitudes'],hedge=hedge['amplitudes'],validation=checks,
        prediction_gate=any(gates.values()),gates=gates,mc_direction_stable=stable,paired_moments=moments,
        validation_scenarios=count,status='diagnostics_complete',
        limitations=['Continuous no-dividend conditional model; all diagnostic windows have 20 remaining transitions.',
                      'Frozen original early-stopping set is not an untouched calibration holdout.'])
    write_json(marker,report)
    if args.export and report['prediction_gate']:
        export(project,output,base,fit,geom,rep,common,cfg,report)


def export(project,output,base,fit,geometry,rep,common,cfg,report):
    from market import simulate,reliability
    from prepare_common import write_json,digest
    t=common['time'].astype(float);n,h=t.shape
    # Reproduce original float64 origin clocks, rather than rounded common time.
    import pandas as pd
    episodes=pd.read_csv(base/'episode_days.csv')
    cal_dates=common['return_dates'];lookup={d:i for i,d in enumerate(common['real_dates'])}
    origins=[g for _,g in episodes.groupby('episode') if g.iloc[0].split_end==cfg['split_ends'][0]]
    selected=np.random.default_rng(cfg['origin_seed']).integers(len(origins),size=n)
    start=np.empty((n,3));clocks=np.empty_like(t)
    for i,j in enumerate(selected):
        p=origins[j];d=p.date.to_numpy(dtype='datetime64[D]');tt=(d-d[0]).astype(float)/365
        clocks[i,:len(tt)]=tt;clocks[i,len(tt):]=tt[-1]
        start[i]=[np.log(p.iloc[0].spot),*common['real_xi'][lookup[p.iloc[0].date]]]
    fine=np.random.default_rng(cfg['shock_seed']).normal(size=(n,h-1,4,3))
    shocks=fine.reshape(n,h-1,2,2,3).sum(3)/np.sqrt(2)
    market_info=json.loads((base/'C1/market.json').read_text())
    for name in ('stat','hedge'):
        out=output/name;out.mkdir(exist_ok=True)
        marker=out/'market.done.json'
        if marker.exists():
            previous=json.loads(marker.read_text())
            if (previous.get('identity')==report['identity'] and previous.get('amplitudes')==report[name]
                and all((out/n).exists() and digest(out/n)==sha for n,sha in previous.get('outputs',{}).items())):
                continue
        coefficients=calibrated(fit,report[name])
        states,diagnostics=simulate(coefficients,start,clocks,shocks,geometry,2,'reflect')
        minimum_slack=min(float((block@(rep['W']*common['scale']).T-rep['bounds']).min())
            for block in np.array_split(states[:,:,1:].reshape(-1,2),128))
        if not np.isfinite(states).all() or minimum_slack < -1e-7:
            raise AssertionError('Calibrated paths fail full-grid factor inequalities')
        diagnostics['full_grid_minimum_slack']=minimum_slack
        # Reliability always uses the reference market, in simulation and observed data.
        q=reliability(states[:,:,1:],clocks,fit,geometry,market_info['initial'])
        q[...,:2]=(np.clip(q[...,:2],market_info['lower'],market_info['upper'])-market_info['center'])/market_info['spread']
        np.savez(out/'single.npz',spot=np.exp(states[:,:,0]).astype('f'),xi=states[:,:,1:].astype('f'),quality=q,
                 member=np.zeros(n,dtype=np.int8))
        shutil.copyfile(base/'C1/common.npz',out/'common.npz')
        write_json(out/'market.done.json',dict(status='complete',identity=report['identity'],
            amplitudes=report[name],diagnostics=diagnostics,outputs={p.name:digest(p) for p in
                [out/'single.npz',out/'common.npz']}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--end',type=int,required=True);p.add_argument('--export',action='store_true')
    p.add_argument('--reference-only',action='store_true')
    run(p.parse_args())
