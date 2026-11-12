"""Atomic, matched generator exports; no policy/evaluation result is read."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from study_core import CONFIG,OUT,PROJECT,HERE,base_factors,digest,write_json
from linear_market import shared_summary

LINEAR=('original_linear','svi_linear','ssvi_linear','essvi_linear')
MODELS=(*LINEAR,'nsde','heston','sabr','localvol')


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _shocks(rng,n,h):
    fine=rng.normal(size=(n,h-1,4,3))
    coarse=fine.reshape(n,h-1,2,2,3).sum(3)/np.sqrt(2)
    return fine,coarse


def _simulate_stochastic(kind,model,p,starts,initial,times,z):
    n,h=times.shape;sub=z.shape[2]
    s=np.empty((n,h));v=np.empty((n,h));s[:,0]=starts;v[:,0]=initial
    rho=model['rho'];xi=model['volvol'] if kind=='heston' else model['nu']
    negatives=0;updates=0
    for j in range(h-1):
        dt=np.maximum(times[:,j+1]-times[:,j],0)/sub;spot=s[:,j].copy();state=v[:,j].copy()
        for a in range(sub):
            noise=z[:,j,a,0];factor=rho*noise+np.sqrt(1-rho*rho)*z[:,j,a,1]
            vol=np.sqrt(state) if kind=='heston' else state
            spot*=np.exp((p['spot_drift']-.5*vol*vol)*dt+vol*np.sqrt(dt)*noise)
            if kind=='heston':
                candidate=state+p['state_kappa']*(p['state_theta']-state)*dt+xi*np.sqrt(state*dt)*factor
                negatives+=int(np.count_nonzero((candidate<0)&(dt>0)));updates+=int(np.count_nonzero(dt>0))
                state=np.maximum(candidate,0)
            else:
                drift=p.get('state_log_drift',p.get('state_drift',0)-.5*xi*xi)
                state*=np.exp(drift*dt+xi*np.sqrt(dt)*factor)
        s[:,j+1]=spot;v[:,j+1]=state
    return s,v,{'negative_variance_truncations':negatives,'active_variance_updates':updates,
                'negative_variance_truncation_fraction':negatives/max(updates,1)}


def _origins(episodes,cfg,common):
    origins=[p for _,p in episodes.groupby('episode',sort=True) if int(p.iloc[0].split_end)==int(cfg['split_ends'][0])]
    n=CONFIG['simulation_train']+CONFIG['simulation_valid']
    if n!=cfg['simulation_train']+cfg['simulation_valid'] or cfg['origin_seed']!=104 or cfg['shock_seed']!=105:
        raise ValueError('frozen sample budget/origin/shock seed mismatch')
    if len(common['time'])!=n or not origins:
        raise ValueError('shared common rows do not match full frozen sample budget')
    selected=np.random.default_rng(104).integers(len(origins),size=n)
    starts=np.array([float(origins[j].iloc[0].spot) for j in selected])
    dates=np.array([str(origins[j].iloc[0].date) for j in selected])
    for i,j in enumerate(selected):
        path=origins[j];a=path.iloc[0]
        t=(path.date.to_numpy(dtype='datetime64[D]')-np.datetime64(a.date)).astype(float)/365
        expected=np.r_[t,np.full(common['time'].shape[1]-len(t),t[-1])]
        if not np.allclose(expected,common['time'][i],atol=1e-7,rtol=0):
            raise ValueError('selected origin time is not matched to shared common inputs')
        if abs(float(a.strike)-common['strike'][i])>1e-8 or abs(float(a.option_close)-common['premium'][i])>1e-8:
            raise ValueError('selected origin strike/premium mismatch')
    if 'gbm' in common and not np.allclose(starts,common['gbm'][:,0],atol=1e-6,rtol=0):
        raise ValueError('origin start spot mismatch')
    return starts,dates,selected


def _pack(spot,info,common):
    t=common['time'];tau=t[:,-1:]-t
    returns=np.diff(np.log(spot),axis=1)
    past=np.concatenate([common['warm'],returns],axis=1)
    rv=np.sqrt(252*np.mean(np.lib.stride_tricks.sliding_window_view(past,20,axis=1)**2,axis=-1))
    base=np.stack([20*np.log(spot/common['strike'][:,None]),tau*12,rv*4],axis=-1)
    f=np.concatenate([base[:,:-1],info[:,:-1]],axis=-1).astype('f')
    if f.shape[-1]!=7 or not np.isfinite(f).all() or not np.isfinite(spot).all() or np.any(spot<=0):
        raise ValueError('nonfinite or malformed seven-feature export')
    return dict(f=f,spot=spot.astype('f'),unit=np.ones_like(spot,dtype='f'),
                active=np.diff(t,axis=1)>0,premium=common['premium'].astype('f'),
                strike_end=common['strike'].astype('f'))


def _state_summary(kind,model,states,ts,ks):
    from stochastic_volatility import price_model
    flat=states.ravel();out=np.empty((len(flat),4));zero_count=int(np.count_nonzero(flat==0))
    if np.any(flat<0):raise ValueError('negative state passed to Q pricing')
    # At v=0 the CIR boundary price is the continuous v->0 limit.
    for a in range(0,len(flat),512):
        state=np.maximum(flat[a:a+512],1e-12)
        calls=price_model(kind,model,ts[None,:,None],ks[None,None,:],state[:,None,None])
        out[a:a+len(state)]=shared_summary(calls,ts,ks)
    return out.reshape(*states.shape,4),zero_count


def _local_spot_drift(cfg,rep,parent):
    from study_core import bootstrap_legacy
    bootstrap_legacy()
    from contract_ledger import EVENTS
    source=parent(PROJECT/'工作区/数据/原始数据/50ETF/510050_华夏上证50ETF.csv')
    calendar=pd.read_csv(source,encoding='gb18030').set_index('交易日期')['收盘价(元)'].sort_index()
    dates=rep['dates'];indices=calendar.index.get_indexer(dates)
    dt=np.diff(dates.astype('datetime64[D]')).astype(float)/365
    legal=(dates[1:]<str(cfg['split_ends'][0]+1))&(indices[:-1]>=0)&(np.diff(indices)==1)
    legal&=~np.isin(dates[1:],[event[0] for event in EVENTS])
    values=calendar.reindex(dates).to_numpy();r=np.diff(np.log(values))[legal];dt=dt[legal]
    if len(r)<3 or not np.isfinite(r).all() or np.any(dt<=0):
        raise ValueError('insufficient legal training spot transitions for localvol drift')
    log_mu=float(r.sum()/dt.sum());variance=float(np.mean((r-log_mu*dt)**2/dt))
    return {'spot_drift':log_mu+.5*variance},dict(formula='sum(logreturn)/sum(ACT365dt) + .5*mean((r-logmu*dt)^2/dt)',
            transitions=len(r),training_before=str(cfg['split_ends'][0]+1),last_training_date=str(dates[1:][legal][-1]),
            excluded='nonadjacent calendar indices, missing dates, dividend ex-dates',log_drift=log_mu,empirical_variance=variance),calendar


def _local_origin_surface(day,calendar,warm,tenors,parent):
    """Registered causal stale<=5 trading days, otherwise origin RV20 flat."""
    position=calendar.index.get_loc(day)
    for age in range(CONFIG['stale_days']+1):
        if position-age<0:break
        candidate=str(calendar.index[position-age]);path=OUT/f'surfaces/essvi/{candidate}/full.json'
        if not path.exists():continue
        record=_json(parent(path))
        if record.get('status')=='complete' and record.get('model',{}).get('diagnostics',{}).get('valid',False):
            return record['model'],dict(fallback_source='same_day' if age==0 else 'past_qualified_curve',
                    source_date=candidate,stale_days=age,origin_date=day)
    if len(warm)!=20 or not np.isfinite(warm).all():raise ValueError('invalid causal RV20 fallback warm history')
    rv=float(np.sqrt(252*np.mean(np.asarray(warm)**2)))
    if rv<=0:raise ValueError('zero RV20 cannot define strictly positive Dupire variance')
    surface={'kind':'essvi','times':list(map(float,tenors)),
             'parameters':[[rv*rv*float(t),0.,0.] for t in tenors]}
    return surface,dict(fallback_source='origin_past20_RV_flat',source_date=day,stale_days=None,
                        origin_date=day,rv20=rv,definition='theta=RV20^2*T,rho=0,psi=0; shared historical warm20')


def _lv_simulate_chunk(tables,dates,starts,t,z,p):
    from local_volatility import _interpolator,_variance
    n,h=t.shape;spot=np.empty((n,h));spot[:,0]=starts;outside=0
    for day in np.unique(dates):
        indices=np.flatnonzero(dates==day);table=tables[day];interpolation=_interpolator(table)
        for j in range(h-1):
            dt=np.maximum(t[indices,j+1]-t[indices,j],0)/z.shape[2]
            current=spot[indices,j].copy()
            for sub in range(z.shape[2]):
                clocks=t[indices,j]+sub*dt
                ratio=current/starts[indices]
                # Vectorized varying per-path clocks, constant boundary extension audited.
                bounded=np.clip(ratio,table['spot_axis'][0],table['spot_axis'][-1])
                outside+=int(np.count_nonzero((bounded!=ratio)&(dt>0)))
                variance=interpolation(np.column_stack([clocks,np.log(bounded)]))
                current*=np.exp((p['spot_drift']-.5*variance)*dt+np.sqrt(variance*dt)*z[indices,j,sub,0])
            spot[indices,j+1]=current
    if outside:raise ValueError(f'localvol paths left frozen coefficient domain: {outside} updates')
    return spot


def _lv_summary(tables,dates,spot,starts,t,ts,ks):
    import local_volatility as lv
    from volatility_surfaces import implied_vol
    info=np.empty((*spot.shape,4));pde_solves=0;max_residual=0.;max_iv_error=0.;refinements=0;wall=time.perf_counter()
    for day in np.unique(dates):
        indices=np.flatnonzero(dates==day);table=tables[day]
        ratios=spot[indices]/starts[indices,None];clock=t[indices]
        calls=np.empty((*ratios.shape,len(ts),len(ks)))
        # A fixed shared strike grid avoids one PDE solve per simulated path.
        strike_grid=table['spot_axis'][::2]
        for ti,tenor in enumerate(ts):
            for elapsed in np.unique(clock):
                loc=np.where(clock==elapsed);s=ratios[loc];query_k=s[:,None]*ks
                if query_k.min()<strike_grid[0] or query_k.max()>strike_grid[-1]:
                    raise ValueError('conditional localvol query strike outside shared PDE grid')
                actual=float(elapsed)
                if actual+tenor>table['times'][-1] and actual+tenor-table['times'][-1]<1e-7:
                    actual=table['times'][-1]-tenor
                # Fixed first-path exact-strike check audits the shared K interpolation.
                direct=lv.conditional_prices(table,actual,float(s[0]),float(tenor),query_k[0])['prices']/s[0]
                for refinement in range(2):
                    grid=strike_grid if refinement==0 else table['spot_axis']
                    result=lv.conditional_prices(table,actual,1.,float(tenor),grid)
                    if not result['diagnostics']['valid']:raise ValueError('localvol shared PDE grid failed call bounds')
                    interpolation=RegularGridInterpolator((result['spot_grid'],grid),result['price_grid'],bounds_error=True)
                    coordinates=np.column_stack([np.repeat(s,len(ks)),query_k.ravel()])
                    values=interpolation(coordinates).reshape(len(s),len(ks))/s[:,None];pde_solves+=1
                    price_error=float(np.max(np.abs(values[0]-direct)))
                    iv_a=implied_vol(float(tenor),ks,values[0]);iv_b=implied_vol(float(tenor),ks,direct)
                    valid_iv=np.isfinite(iv_a)&np.isfinite(iv_b)
                    iv_error=float(np.max(np.abs(iv_a[valid_iv]-iv_b[valid_iv]))) if np.any(valid_iv) else 0.
                    if price_error<=1e-4 and iv_error<=.005:break
                    if refinement==0:refinements+=1
                else:
                    raise ValueError(f'localvol numerical_failed after one refinement: price={price_error}, IV={iv_error}')
                calls[loc[0],loc[1],ti,:]=values
                max_residual=max(max_residual,price_error);max_iv_error=max(max_iv_error,iv_error)
        info[indices]=shared_summary(calls,ts,ks)
        lv._PDE_CACHE.clear() # Origin complete; prevent multi-origin grid-cache memory growth.
    elapsed=time.perf_counter()-wall
    return info,dict(pde_solves=pde_solves,seconds=elapsed,shared_strike_grid_nodes=len(strike_grid),
                     fixed_query_interpolation_max_error=max_residual,
                     fixed_query_iv_max_error=max_iv_error,once_only_refinements=refinements,
                     cost_estimate='actual elapsed for all selected origin/clock groups; budget not reduced')


def export_generator(end,model):
    """Export exactly CONFIG train+valid rows; CLI never substitutes a pilot."""
    root=OUT/f'inputs/{end}';root.mkdir(parents=True,exist_ok=True)
    ready=root/f'sim_{model}.ready.json';target=root/f'sim_{model}.npz';tic=time.perf_counter();parents={}
    # A stale ready receipt cannot advertise a newly failing/rebuilt artifact.
    if ready.exists():ready.unlink()
    try:
        if model not in MODELS:raise ValueError('unsupported generator model')
        old=PROJECT/f'工作区/数据/处理后/创新组合实验/{end}'
        def parent(path):
            path=Path(path);parents[str(path)]=digest(path);return path
        cfg=_json(parent(old/'config.json'))
        common=dict(np.load(parent(old/'C1/common.npz'),allow_pickle=False))
        episodes=pd.read_csv(parent(old/'episode_days.csv'))
        rep=dict(np.load(parent(base_factors(end)),allow_pickle=False))
        starts,dates,selected=_origins(episodes,cfg,common);n,h=common['time'].shape
        episode_ids=np.array([int(p.iloc[0].episode) for _,p in episodes.groupby('episode',sort=True)
                              if int(p.iloc[0].split_end)==int(cfg['split_ends'][0])])[selected]
        sidecar_arrays=dict(origin_dates=dates,episode_id=episode_ids,selected_origin_index=selected)
        ts_all=np.unique(rep['grid_t']);ks=np.unique(rep['grid_k'])
        bracket=int(np.clip(np.searchsorted(ts_all,30/365)-1,0,len(ts_all)-2));ts=ts_all[bracket:bracket+2]
        diagnostics={};directory=OUT/f'markets/{end}/{model}'
        if model in LINEAR or model=='nsde':
            status=_json(parent(directory/'market.json'))
            if status.get('status')!='complete':raise ValueError('linear market fit is not complete')
            if model=='nsde':
                ident=status.get('identity',{})
                if ident.get('head_widths')!=[64,64,7] or ident.get('market_weights_reused') is not False:
                    raise ValueError('G1 requires a fresh nonlinear Dense64/Dense64/Dense7 reference fit')
                if ident.get('source')!=digest(HERE/'reference_market.py'):
                    raise ValueError('fresh NSDE reference source identity mismatch')
                if status.get('output_sha256',{}).get('simulation.npz')!=digest(directory/'simulation.npz'):
                    raise ValueError('fresh NSDE reference simulation SHA mismatch')
            data=dict(np.load(parent(directory/'simulation.npz'),allow_pickle=False))
            for name in ('time','warm','strike','premium'):
                if data[name].shape!=common[name].shape or not np.allclose(data[name],common[name],atol=1e-7,rtol=0):
                    raise ValueError('linear shared input mismatch: '+name)
            spot=data['spot'];info=data['summary']
            if data['xi'].shape[:2]!=(n,h):raise ValueError('linear diagnostic xi sidecar shape mismatch')
            sidecar_arrays['xi']=data['xi']
            if spot.shape!=(n,h) or info.shape!=(n,h,4) or not np.allclose(spot[:,0],starts,atol=1e-6,rtol=0):
                raise ValueError('linear simulation shape/origin mismatch')
            diagnostics['source']='fresh nonlinear constrained NSDE reference' if model=='nsde' else 'completed low-capacity market simulation'
        else:
            from stochastic_volatility import price_model
            if model=='localvol':
                from local_volatility import build_table
                p,drift_diagnostics,calendar=_local_spot_drift(cfg,rep,parent)
                diagnostics['localvol_spot_drift']=drift_diagnostics
                tables={};fallbacks={}
                for day in np.unique(dates):
                    warm=common['warm'][np.flatnonzero(dates==day)[0]]
                    surface,fallbacks[day]=_local_origin_surface(day,calendar,warm,ts_all,parent)
                    # A selected qualified curve's genuine Dupire error is not
                    # hidden by stale/RV replacement; build_table must succeed.
                    tables[day]=build_table(surface,CONFIG['localvol_horizon_days'])
                diagnostics['origin_fallbacks']=fallbacks
                q=None;initial=None
            else:
                status=_json(parent(directory/'status.json'))
                if not status.get('complete'):raise ValueError('stochastic calibration is not complete')
                q=_json(parent(directory/'global.json'))['model']
                if not q['diagnostics']['valid']:raise ValueError('global Q numerical diagnostics failed')
                p=_json(parent(directory/'p_parameters.json'))
                states=pd.read_csv(parent(directory/'states.csv')).set_index('date')['state']
                initial=states.reindex(dates).to_numpy()
                if not np.isfinite(initial).all():raise ValueError('missing causal origin state')
            spot=np.empty((n,h));latent=np.empty((n,h)) if model!='localvol' else None
            rng=np.random.default_rng(105);negative=updates=0;paired=[];paired_state=[]
            for a in range(0,n,512):
                b=min(a+512,n);fine,coarse=_shocks(rng,b-a,h);t=common['time'][a:b]
                pair=np.flatnonzero(np.arange(a,b)%8==0)
                if model=='localvol':
                    coarse_s=_lv_simulate_chunk(tables,dates[a:b],starts[a:b],t,coarse,p)
                    fine_s=_lv_simulate_chunk(tables,dates[a:b][pair],starts[a:b][pair],t[pair],fine[pair],p)
                else:
                    coarse_s,state,d=_simulate_stochastic(model,q,p,starts[a:b],initial[a:b],t,coarse)
                    fine_s,fstate,_=_simulate_stochastic(model,q,p,starts[a:b][pair],initial[a:b][pair],t[pair],fine[pair])
                    latent[a:b]=state;negative+=d['negative_variance_truncations'];updates+=d['active_variance_updates']
                    paired_state.extend((state[pair,-1]-fstate[:,-1]).tolist())
                spot[a:b]=coarse_s
                paired.extend((np.log(coarse_s[pair,-1])-np.log(fine_s[:,-1])).tolist())
            diagnostics.update(negative_variance_truncations=negative,active_variance_updates=updates,
                negative_variance_truncation_fraction=negative/max(updates,1),
                paired_four_substep_paths=len(paired),paired_terminal_logspot_rmse=float(np.sqrt(np.mean(np.square(paired)))))
            if paired_state:
                diagnostics['paired_terminal_state_rmse']=float(np.sqrt(np.mean(np.square(paired_state))))
            if model=='localvol':
                info,lv_diag=_lv_summary(tables,dates,spot,starts,common['time'],ts,ks)
                diagnostics['localvol']=lv_diag
                diagnostics['tables']={day:dict(surface_hash=t['surface_hash'],diagnostics=t['diagnostics']) for day,t in tables.items()}
                sidecar_arrays['origin_surface_hash']=np.array([tables[day]['surface_hash'] for day in dates])
            else:
                from stochastic_volatility import pricing_diagnostics
                pricing_diagnostics(reset=True)
                info,zeros=_state_summary(model,q,latent,ts,ks);diagnostics['zero_boundary_pricing_count']=zeros
                sample=latent.ravel()[np.unique(np.linspace(0,latent.size-1,min(16,latent.size)).astype(int))]
                full=price_model(model,q,ts_all[None,:,None],ks[None,None,:],np.maximum(sample,1e-12)[:,None,None])
                reduced,_=_state_summary(model,q,sample,ts,ks)
                diagnostics['full_20_grid_sample_summary_max_difference']=float(np.max(np.abs(shared_summary(full,ts_all,ks)-reduced)))
                if model=='heston':diagnostics['pricing_quadrature']=pricing_diagnostics()
                sidecar_arrays['latent']=latent.astype('f')
        arrays=_pack(spot,info,common)
        sources={name:digest(HERE/name) for name in ['generator_export.py','study_core.py','linear_market.py',
                    'stochastic_volatility.py','local_volatility.py','volatility_surfaces.py']}
        if model=='nsde':sources['reference_market.py']=digest(HERE/'reference_market.py')
        identity=dict(model=model,end=int(end),parents=parents,source=digest(HERE/'generator_export.py'),sources=sources,
                      config=CONFIG,origin_seed=104,shock_seed=105,origin_order='pandas numeric episode groupby sort=True',
                      rows=n,train=CONFIG['simulation_train'],valid=CONFIG['simulation_valid'],
                      feature_slots='base3+shared_summary4; prior position appended by policy as eighth slot')
        temporary=target.with_suffix('.npz.tmp')
        with temporary.open('wb') as handle:np.savez_compressed(handle,**arrays)
        temporary.replace(target)
        sidecar=root/f'sim_{model}.state_sidecar.npz';temporary=sidecar.with_suffix('.npz.tmp')
        with temporary.open('wb') as handle:np.savez_compressed(handle,**sidecar_arrays)
        temporary.replace(sidecar)
        receipt=dict(complete=True,model=model,end=int(end),rows=n,input_sha256=digest(target),
                     state_sidecar=dict(file=sidecar.name,sha256=digest(sidecar),complete=True),
                     source=identity['source'],parents=parents,identity=identity,diagnostics=diagnostics,
                     seconds=time.perf_counter()-tic)
        write_json(ready,receipt)
        return receipt
    except Exception as error:
        numerical=model=='localvol' and any(token in str(error) for token in ('Dupire','PDE','numerical_failed'))
        write_json(root/f'sim_{model}.failure.json',dict(complete=False,model=model,end=int(end),
                   status='numerical_failed' if numerical else 'failed',
                   source=digest(HERE/'generator_export.py'),config=CONFIG,
                   error_type=type(error).__name__,error=str(error),parents=parents,seconds=time.perf_counter()-tic))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--end',type=int,required=True)
    parser.add_argument('--model',choices=MODELS,required=True)
    arguments=parser.parse_args();export_generator(arguments.end,arguments.model)
