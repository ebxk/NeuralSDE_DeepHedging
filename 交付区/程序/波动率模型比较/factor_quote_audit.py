"""Independent masked-quote states with frozen two-factor geometry.

Original repaired 20-grid is a static reference, not another generator.
"""
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize,linprog
from study_core import CONFIG,OUT,PROJECT,HERE,SHARED_SURFACES,base_factors,bootstrap_legacy,digest,write_json


def _curve(t,k,xi,rep):
    bootstrap_legacy()
    from extended_curve import curve
    return curve(np.asarray(t,float),np.asarray(k,float),np.asarray(xi,float),rep,np.ones(2))


def clean_quotes(quotes):
    """One effective date/expiry/strike, never a duplicate across masks."""
    q=quotes.copy()
    if not pd.api.types.is_bool_dtype(q.holdout):
        mapped=q.holdout.astype(str).str.lower().map({'true':True,'false':False})
        if mapped.isna().any():raise ValueError('holdout must be a true/false mask')
        q['holdout']=mapped.astype(bool)
    keys=['date','expire_date','strike'] if {'expire_date','strike'}.issubset(q.columns) else ['date','ttm','k']
    duplicates=q.duplicated(keys,keep=False)
    for _,g in q[duplicates].groupby(keys,sort=False):
        if g.holdout.nunique()!=1 or np.ptp(g.price.to_numpy())>1e-10:
            raise ValueError('duplicate effective quote has conflicting target or crosses visible/heldout masks')
    q=q.drop_duplicates(keys,keep='first').reset_index(drop=True)
    values=q[['ttm','k','price']].to_numpy(float)
    if not len(q) or not np.isfinite(values).all() or np.any(values[:,:2]<=0):
        raise ValueError('positive finite quote time/strike and finite price required')
    if np.any(q.price.to_numpy()<np.maximum(1-q.k.to_numpy(),0)-1e-10) or np.any(q.price.to_numpy()>=1):
        raise ValueError('quote outside normalized call bounds')
    return q


def training_starts(rep,training_before,seed=20261112):
    mask=np.asarray(rep['dates']).astype(str)<str(training_before)
    pool=np.asarray(rep['xi'],float)[mask]
    if not len(pool) or pool.shape[1:]!=(2,) or not np.isfinite(pool).all():
        raise ValueError('missing legal training-only two-factor initial states')
    indices=np.random.default_rng(seed).choice(len(pool),size=5,replace=len(pool)<5)
    return pool[indices].copy()


def _view_mask(q,view):
    if view=='full':return np.ones(len(q),bool)
    if view=='heldout':return ~q.holdout.to_numpy()
    raise ValueError('view must be full or heldout')


def _metrics(q,prediction,fit_mask,rep):
    error=np.asarray(prediction)-q.price.to_numpy();visible=~q.holdout.to_numpy();hidden=~visible
    ts=np.unique(rep['grid_t']);ks=np.unique(rep['grid_k'])
    outside=(q.ttm.to_numpy()<ts[0])|(q.ttm.to_numpy()>ts[-1])|(q.k.to_numpy()<ks[0])|(q.k.to_numpy()>ks[-1])
    return dict(visible_n=int(visible.sum()),heldout_n=int(hidden.sum()),fit_n=int(fit_mask.sum()),
                visible_rmse=float(np.sqrt(np.mean(error[visible]**2))) if visible.any() else None,
                heldout_rmse=float(np.sqrt(np.mean(error[hidden]**2))) if hidden.any() else None,
                visible_sse=float(np.sum(error[visible]**2)),heldout_sse=float(np.sum(error[hidden]**2)),
                objective=float(np.mean(error[fit_mask]**2)),out_of_grid_n=int(outside.sum()),
                out_of_grid_fraction=float(np.mean(outside)),
                heldout_is_out_of_sample=bool(not np.any(fit_mask&hidden)))


def fit_factor_view(rep,quotes,starts,view):
    """Frozen geometry; full fit descriptive, heldout fit excludes every mask."""
    tic=time.perf_counter();q=clean_quotes(quotes);fit_mask=_view_mask(q,view)
    if not fit_mask.any():raise ValueError('no visible quotes for independent state fitting')
    starts=np.asarray(starts,float)
    if starts.shape!=(5,2):raise ValueError('exactly five training-only initial states required')
    t=q.ttm.to_numpy();k=q.k.to_numpy();target=q.price.to_numpy();w=np.asarray(rep['W']);bounds=np.asarray(rep['bounds'])
    def value_gradient(x):
        prediction,_,df=_curve(t[fit_mask],k[fit_mask],np.broadcast_to(x,(int(fit_mask.sum()),2)),rep)
        residual=prediction-target[fit_mask]
        return 1e4*float(np.mean(residual**2)),1e4*2*np.mean(residual[:,None]*df,axis=0)
    attempts=[];candidates=[]
    for initial in starts:
        result=minimize(value_gradient,initial.copy(),jac=True,method='SLSQP',
            constraints={'type':'ineq','fun':lambda x:w@x-bounds,'jac':lambda x:w},
            options={'maxiter':300,'ftol':1e-12})
        slack=float(np.min(w@result.x-bounds))
        success=bool(result.success and slack>=-1e-7 and np.isfinite(result.fun))
        attempts.append(dict(success=success,optimizer_success=bool(result.success),slack=slack,
                             objective=float(result.fun/1e4),message=str(result.message)))
        candidates.append((not success,float(result.fun),result.x,slack))
    failed,_,xi,slack=min(candidates,key=lambda item:item[:2])
    prediction,_,_=_curve(t,k,np.broadcast_to(xi,(len(q),2)),rep)
    metrics=_metrics(q,prediction,fit_mask,rep)
    return dict(view=view,valid=bool(not failed and np.isfinite(prediction).all()),xi=xi.tolist(),
                constraint_min=slack,attempts=attempts,seconds=time.perf_counter()-tic,
                price_scope='legacy_extended_curve_finite_grid_and_documented_extrapolation',**metrics)


def repair_grid(raw,a,b):
    raw=np.asarray(raw,float);a=np.asarray(a,float);b=np.asarray(b,float)
    slack=float(np.min(a@raw-b));fixed=raw.copy();repaired=slack<-1e-9
    if repaired:
        n=len(raw)
        result=linprog(np.ones(2*n),A_ub=np.c_[-a,a],b_ub=a@raw-b,bounds=(0,None),method='highs')
        if not result.success:raise ValueError('original grid L1 repair failed: '+result.message)
        fixed=raw+result.x[:n]-result.x[n:]
    minimum=float(np.min(a@fixed-b))
    if minimum<-1e-7:raise ValueError('original grid remains financially infeasible')
    return fixed,dict(repaired=bool(repaired),raw_constraint_min=slack,constraint_min=minimum,
                      repair_l1=float(np.sum(np.abs(fixed-raw))),repair_mean_abs=float(np.mean(np.abs(fixed-raw))))


def fit_original_view(quotes,grid_t,grid_k,a,b,view):
    """Original market_surfaces interpolation/anchors plus original A,b L1."""
    tic=time.perf_counter();q=clean_quotes(quotes);fit_mask=_view_mask(q,view)
    visible=q[fit_mask]
    if visible.empty:raise ValueError('no visible quote for original grid')
    ts=np.unique(grid_t);ks=np.unique(grid_k);intrinsic=np.maximum(1-ks,0)
    times=[0.];curves=[intrinsic]
    strike_extrapolated=0
    for tenor,g in visible.groupby('ttm',sort=True):
        g=g.sort_values('k');money=np.r_[0.,g.k.to_numpy()];calls=np.r_[1.,g.price.to_numpy()]
        value=np.interp(ks,money,calls);right=ks>money[-1]
        # Exactly the original finite-grid constructor's bounded -0.5 wing.
        value[right]=np.maximum(intrinsic[right],calls[-1]-.5*(ks[right]-money[-1]))
        strike_extrapolated+=int(np.count_nonzero((ks<g.k.min())|right))
        times.append(float(tenor));curves.append(value)
    raw=np.array([np.interp(ts,times,col) for col in np.asarray(curves).T]).T.ravel()
    fixed,repair=repair_grid(raw,a,b)
    rep=dict(grid_t=np.asarray(grid_t),grid_k=np.asarray(grid_k),center=fixed,basis=np.zeros((2,len(fixed))))
    prediction,_,_=_curve(q.ttm.to_numpy(),q.k.to_numpy(),np.zeros((len(q),2)),rep)
    metrics=_metrics(q,prediction,fit_mask,rep)
    return dict(view=view,valid=bool(np.isfinite(prediction).all()),grid=fixed.tolist(),raw_grid=raw.tolist(),
                seconds=time.perf_counter()-tic,repair=repair,constraint_min=repair['constraint_min'],
                construction_strike_extrapolated_nodes=strike_extrapolated,
                construction_tenor_extrapolated_nodes=int(np.count_nonzero(ts>times[-1]))*len(ks),
                price_scope='original_repaired_finite_grid_reference',**metrics)


def _split(day,fold):
    if day<str(fold[0]+1):return 'training'
    if day<str(fold[1]+1):return 'validation'
    return 'historical_evaluation'


def run(end,representation):
    root=OUT/f'factor_quote_audit/{end}/{representation}';root.mkdir(parents=True,exist_ok=True);parents={}
    def parent(path):
        path=type(root)(path);parents[str(path)]=digest(path);return path
    try:
        if representation not in ('nsde','svi','ssvi','essvi','original'):raise ValueError('unknown representation')
        cfg_path=parent(PROJECT/f'工作区/数据/处理后/创新组合实验/{end}/config.json')
        cfg=json.loads(cfg_path.read_text(encoding='utf-8-sig'));fold=cfg['split_ends']
        raw_quotes=pd.read_csv(parent(OUT/'data/quotes.csv'))
        quotes=clean_quotes(raw_quotes[raw_quotes.date<str(end+1)])
        if representation=='original':
            source=parent(SHARED_SURFACES)
            with np.load(source,allow_pickle=False) as data:
                original={key:data[key] for key in ('grid_t','grid_k','A','b')}
            starts=None;start_identity=None
        else:
            source=parent(base_factors(end) if representation=='nsde'
                          else OUT/f'representations/{end}/{representation}_linear.npz')
            rep=dict(np.load(source,allow_pickle=False));starts=training_starts(rep,str(fold[0]+1),CONFIG['calibration_seed'])
            start_identity=hashlib.sha256(starts.tobytes()).hexdigest()
        bootstrap_legacy()
        import extended_curve
        parent(extended_curve.__file__)
        identity=dict(end=int(end),representation=representation,parents=parents,source=digest(HERE/'factor_quote_audit.py'),
                      fold=fold,seed=CONFIG['calibration_seed'],training_before=str(fold[0]+1),
                      initialization_sha256=start_identity,starts=starts.tolist() if starts is not None else None,
                      quote_policy='visible/heldout disjoint effective strike; independent masked state',
                      original_reference_is_generator=False)
        write_json(root/'identity.json',identity)
        write_json(root/'status.json',dict(complete=False,status='running',identity=identity))
        identity_hash=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        rows=[];failures=0
        temporary=root/'daily.jsonl.tmp'
        with temporary.open('w',encoding='utf8') as handle:
            for day,q in quotes.groupby('date',sort=True):
                for view in ('full','heldout'):
                    if representation=='original':
                        result=fit_original_view(q,original['grid_t'],original['grid_k'],original['A'],original['b'],view)
                    else:result=fit_factor_view(rep,q,starts,view)
                    result.update(date=day,split=_split(day,fold),identity_sha256=identity_hash)
                    failures+=int(not result['valid']);handle.write(json.dumps(result,allow_nan=False)+'\n');handle.flush()
                    rows.append({key:value for key,value in result.items() if key not in ('attempts','grid','raw_grid','repair','xi')})
        temporary.replace(root/'daily.jsonl')
        frame=pd.DataFrame(rows);temporary=root/'daily.csv.tmp';frame.to_csv(temporary,index=False);temporary.replace(root/'daily.csv')
        summary=[]
        for (split,view),g in frame.groupby(['split','view'],sort=False):
            visible_n=int(g.visible_n.sum());hidden_n=int(g.heldout_n.sum())
            summary.append(dict(split=split,view=view,days=len(g),failed_days=int((~g.valid).sum()),
                visible_n=visible_n,heldout_n=hidden_n,
                visible_rmse=float(np.sqrt(g.visible_sse.sum()/visible_n)) if visible_n else None,
                heldout_rmse=float(np.sqrt(g.heldout_sse.sum()/hidden_n)) if hidden_n else None,
                heldout_is_out_of_sample=view=='heldout' and split!='training',
                interpretation='training-internal frozen-geometry holdout' if split=='training' else 'fixed cross-sectional holdout on observed historical data',
                constraint_min=float(g.constraint_min.min()),out_of_grid_fraction=float(g.out_of_grid_n.sum()/(visible_n+hidden_n))))
        write_json(root/'summary.json',summary)
        receipt=dict(complete=failures==0,status='complete' if failures==0 else 'fit_failed',identity=identity,
                     dates=int(quotes.date.nunique()),failed_fits=failures,scope='finite_grid_and_legacy_extrapolation_only',
                     output_sha256={name:digest(root/name) for name in ('daily.jsonl','daily.csv','summary.json')})
        write_json(root/'status.json',receipt);return receipt
    except Exception as error:
        write_json(root/'status.json',dict(complete=False,status='failed',error_type=type(error).__name__,
                   error=str(error),parents=parents,source=digest(HERE/'factor_quote_audit.py')))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--end',required=True,type=int)
    parser.add_argument('--representation',required=True,choices=['nsde','svi','ssvi','essvi','original'])
    args=parser.parse_args();run(args.end,args.representation)
