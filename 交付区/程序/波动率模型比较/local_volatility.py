"""Frozen-origin eSSVI Dupire table, local-vol paths and conditional Q PDE.

Times and spot/strike ratios are relative to the origin surface, never reset
to the observation date. No future surface, file or quote is read.
"""
import hashlib
import json
import numpy as np
from scipy.interpolate import RegularGridInterpolator, CubicHermiteSpline
from scipy.linalg import solve_banded
from volatility_surfaces import total_variance, price_surface

_PDE_CACHE={}


def _surface_hash(surface):
    keys=('kind','times','parameters','rho','eta','gamma')
    payload={key:surface[key] for key in keys if key in surface}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def build_table(surface,horizon_days=120):
    """Daily Dupire table on 401 log-spaced spot ratios [.2,3.5].

    Direct 2*C_t/(K^2*C_KK) is used only when a well-resolved denominator
    agrees with the equivalent stable total-variance formula w_t/g to 1e-5.
    Unresolved tails use w_t/g; negative w_t or nonpositive g raises rather
    than clipping. The t=0 row uses the limit approximated at 1e-5 years.
    """
    if surface.get('kind')!='essvi' or horizon_days<=0:
        raise ValueError('positive horizon and frozen eSSVI surface required')
    times=np.linspace(0,horizon_days/365,int(np.ceil(horizon_days))+1)
    spots=np.exp(np.linspace(np.log(.2),np.log(3.5),401))
    t=np.maximum(times[:,None],1e-5);k=spots[None,:];x=np.log(k)
    ht=np.minimum(1e-5,t/4);hx=1e-3
    w=total_variance(surface,t,k)
    wp=(total_variance(surface,t,k*np.exp(hx))-total_variance(surface,t,k*np.exp(-hx)))/(2*hx)
    wpp=(total_variance(surface,t,k*np.exp(hx))-2*w+total_variance(surface,t,k*np.exp(-hx)))/hx**2
    wt=(total_variance(surface,t+ht,k)-total_variance(surface,t-ht,k))/(2*ht)
    if np.any(w<=0):
        raise ValueError('Dupire undefined: nonpositive total variance')
    g=(1-x*wp/(2*w))**2-wp**2/4*(1/w+.25)+wpp/2
    if not np.isfinite([w,wt,g]).all() or np.any(g<=1e-8) or np.any(wt<=0):
        raise ValueError('Dupire invalid: nonpositive variance derivative or density denominator; no clipping applied')
    stable=wt/g
    hk=2e-4*k
    c=price_surface(surface,t,k)
    ct=(price_surface(surface,t+ht,k)-price_surface(surface,t-ht,k))/(2*ht)
    ckk=(price_surface(surface,t,k+hk)-2*c+price_surface(surface,t,k-hk))/hk**2
    denominator=k*k*ckk
    direct=np.divide(2*ct,denominator,out=np.full_like(stable,np.nan),where=denominator>1e-9)
    reliable=(denominator>1e-9)&np.isfinite(direct)&(direct>0)&(np.abs(direct/stable-1)<=1e-5)
    variance=np.where(reliable,direct,stable)
    if not np.isfinite(variance).all() or np.any(variance<=0):
        raise ValueError('nonpositive/nonfinite local variance; no clipping applied')
    source_hash=_surface_hash(surface)
    content_hash=hashlib.sha256(source_hash.encode()+times.tobytes()+spots.tobytes()+variance.tobytes()).hexdigest()
    diagnostics={'valid':True,'clipped_points':0,'negative_stable_variance_points':0,
                 'tiny_price_denominator_points':int(np.count_nonzero(denominator<=1e-9)),
                 'negative_price_candidate_points':int(np.count_nonzero(np.isfinite(direct)&(direct<0))),
                 'price_branch_fraction':float(np.mean(reliable)),
                 'stable_branch_fraction':float(np.mean(~reliable)),
                 'variance_min':float(variance.min()),'variance_max':float(variance.max()),
                 'g_min':float(g.min()),'origin_time_limit_years':1e-5,
                 'scope':'finite_difference_table_only','price_denominator_floor':1e-9}
    extension=times>=90/365
    if extension.any():
        diagnostics['days90_120']=dict(nodes=int(extension.sum()),
            variance_min=float(variance[extension].min()),variance_max=float(variance[extension].max()),
            variance_mean=float(variance[extension].mean()),g_min=float(g[extension].min()),
            beyond_last_quoted_tenor_fraction=float(np.mean(times[extension]>max(surface['times']))))
    return {'times':times,'spot_axis':spots,'local_variance':variance,
            'surface_hash':source_hash,'table_hash':content_hash,'diagnostics':diagnostics,
            'horizon_days':float(horizon_days),'coordinate_system':'origin_spot_ratios'}


def _interpolator(table):
    if not table['diagnostics']['valid']:
        raise ValueError('invalid local volatility table')
    return RegularGridInterpolator((table['times'],np.log(table['spot_axis'])),table['local_variance'],bounds_error=True)


def _variance(interpolator,t,spots,axis):
    spots=np.asarray(spots,float)
    boundary=(spots<axis[0])|(spots>axis[-1])
    # Explicit boundary extension for path evolution; never silently accepted.
    clipped=np.clip(spots,axis[0],axis[-1])
    coords=np.column_stack([np.full(spots.size,t),np.log(clipped.ravel())])
    return interpolator(coords).reshape(spots.shape),int(np.count_nonzero(boundary))


def simulate_p(table,start_spot,times,shocks,spot_drift=0.):
    """Log-Euler local-vol P paths; shocks shape [...,steps,2], spot in col0.

    start_spot is in origin-spot units. Col1 is deliberately unused for this
    one-factor diffusion. Boundary coefficient extension is counted and makes
    diagnostics.valid=False. It never clips the simulated spot itself.
    """
    z=np.asarray(shocks,float);times=np.asarray(times,float)
    if z.ndim<2 or z.shape[-1]!=2 or times.ndim!=1 or len(times)!=z.shape[-2]+1:
        raise ValueError('times must have steps+1 elements; shocks [...,steps,2]')
    if not np.isfinite(z).all() or not np.isfinite(times).all() or not np.isfinite(spot_drift) or start_spot<=0:
        raise ValueError('finite inputs and positive start spot required')
    if np.any(np.diff(times)<=0) or times[0]<table['times'][0] or times[-1]>table['times'][-1]+1e-12:
        raise ValueError('increasing simulation times must lie within frozen table horizon')
    interpolation=_interpolator(table);shape=z.shape[:-2]+(len(times),)
    spot=np.empty(shape);spot[...,0]=start_spot;outside=0
    for j,dt in enumerate(np.diff(times)):
        var,count=_variance(interpolation,times[j],spot[...,j],table['spot_axis']);outside+=count
        spot[...,j+1]=spot[...,j]*np.exp((spot_drift-.5*var)*dt+np.sqrt(var*dt)*z[...,j,0])
    mean_terminal=float(np.mean(spot[...,-1]));expected=float(start_spot*np.exp(spot_drift*(times[-1]-times[0])))
    outside+=int(np.count_nonzero((spot[...,-1]<table['spot_axis'][0])|(spot[...,-1]>table['spot_axis'][-1])))
    return {'spot':spot,'diagnostics':{'valid':bool(outside==0 and np.isfinite(spot).all()),
            'coefficient_boundary_extension_count':outside,'spot_clipped_count':0,
            'mean_terminal_spot':mean_terminal,'expected_spot_from_drift':expected,
            'mean_error':mean_terminal-expected,'terminal_spot_variance':float(np.var(spot[...,-1])),
            'second_shock_unused':True,'surface_hash':table['surface_hash']}}


def _solve_grid(table,time,tenor,strikes):
    s=table['spot_axis'];x=np.log(s);dx=x[1]-x[0]
    value=np.maximum(s[:,None]-strikes[None,:],0)
    lower=value[0].copy();upper=value[-1].copy()
    if tenor<=0:
        return value,np.gradient(value,s,axis=0,edge_order=2)
    interp=_interpolator(table);steps=max(4,int(np.ceil(tenor*730)));h=tenor/steps
    for step in range(steps):
        clock=time+tenor-(step+.5)*h
        v,_=_variance(interp,clock,s[1:-1],s)
        a=.5*v*(1/dx**2+1/(2*dx));b=-v/dx**2;c=.5*v*(1/dx**2-1/(2*dx))
        weight=1. if step<2 else .5  # Rannacher-style implicit start.
        rhs=value[1:-1]+(1-weight)*h*(a[:,None]*value[:-2]+b[:,None]*value[1:-1]+c[:,None]*value[2:])
        rhs[0]+=weight*h*a[0]*lower;rhs[-1]+=weight*h*c[-1]*upper
        bands=np.zeros((3,len(s)-2));bands[1]=1-weight*h*b
        bands[0,1:]=-weight*h*c[:-1];bands[2,:-1]=-weight*h*a[1:]
        value[1:-1]=solve_banded((1,1),bands,rhs)
    return value,np.gradient(value,s,axis=0,edge_order=2)


def conditional_prices(table,time,spot_ratio,tenor,strike_ratios):
    """Conditional Q calls C/S_origin, Delta and reusable multi-strike grid.

    Strikes remain K/S_origin. time is elapsed since the frozen origin curve;
    tenor is remaining lifetime. No resetting the LV clock or reading curves
    from future observation dates. Returned grids are copies of cached grids.
    """
    strikes=np.atleast_1d(np.asarray(strike_ratios,float));time=float(time);tenor=float(tenor)
    if strikes.ndim!=1 or not np.isfinite(strikes).all() or np.any(strikes<=0):
        raise ValueError('positive finite one-dimensional strikes required')
    if not np.isfinite([time,tenor,spot_ratio]).all() or time<0 or tenor<0 or time+tenor>table['times'][-1]+1e-12:
        raise ValueError('condition and maturity times must lie within frozen table horizon')
    s=table['spot_axis']
    if spot_ratio<s[0] or spot_ratio>s[-1]:
        raise ValueError('conditional spot outside finite PDE grid')
    # Table hash includes source surface identity and all coefficient/grid bytes.
    key=hashlib.sha256((table['table_hash']+repr(time)+repr(tenor)).encode()+strikes.tobytes()).hexdigest()
    cache_hit=key in _PDE_CACHE
    if not cache_hit:
        _PDE_CACHE[key]=_solve_grid(table,time,tenor,strikes)
    grid,delta_grid=_PDE_CACHE[key]
    interpolated=CubicHermiteSpline(s,grid,delta_grid,axis=0)
    prices=interpolated(spot_ratio)
    delta=interpolated(spot_ratio,1)
    if tenor==0:
        prices=np.maximum(spot_ratio-strikes,0)
        delta=np.where(spot_ratio>strikes,1.,np.where(spot_ratio<strikes,0.,.5))
    valid=bool(np.isfinite(prices).all() and np.all(prices>=np.maximum(spot_ratio-strikes,0)-1e-6) and np.all(prices<=spot_ratio+1e-6))
    return {'prices':prices,'delta':delta,'spot_grid':s.copy(),'price_grid':grid.copy(),
            'delta_grid':delta_grid.copy(),'cache_key':key,'surface_hash':table['surface_hash'],
            'diagnostics':{'valid':valid,'cache_hit':cache_hit,'scheme':'CN_implicit_first_two_steps',
                           'max_time_step_years':1/730,'scope':'finite_PDE_grid_only'}}
