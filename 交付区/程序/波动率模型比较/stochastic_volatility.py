"""Separate Q pricing/calibration and P transition estimation/simulation.

All calls are normalized by spot with zero rates/dividends. Public strikes are
K/spot. No function reads files, clocks, future quotes, or random global state.
"""
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar, least_squares
from scipy.special import ndtr
from scipy.integrate import quad

_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(256)
_PRICING={'quotes':0,'fallback_quotes':0,'fallback_max_error_estimate':0.}


def pricing_diagnostics(reset=False):
    result=dict(_PRICING)
    result['fallback_fraction']=result['fallback_quotes']/max(result['quotes'],1)
    if reset:
        _PRICING.update(quotes=0,fallback_quotes=0,fallback_max_error_estimate=0.)
    return result


def _lewis_quad(t,k,v,kappa,theta,xi,rho):
    """Damped Lewis inversion with adaptive Fourier-tail quadrature."""
    logk=np.log(k)
    def integrand(u,imaginary=False):
        value=_heston_cf(u-.5j,t,v,kappa,theta,xi,rho)/(u*u+.25)
        return float(value.imag if imaginary else value.real)
    if abs(logk)<1e-8:
        integral,error=quad(integrand,0,np.inf,epsabs=2e-11,epsrel=2e-10,limit=300)
    else:
        cosine,e1=quad(integrand,0,np.inf,weight='cos',wvar=abs(logk),epsabs=2e-11,limit=300,limlst=300)
        sine,e2=quad(lambda u:integrand(u,True),0,np.inf,weight='sin',wvar=abs(logk),epsabs=2e-11,limit=300,limlst=300)
        integral=cosine+np.sign(logk)*sine;error=e1+e2
    _PRICING['fallback_max_error_estimate']=max(_PRICING['fallback_max_error_estimate'],float(np.sqrt(k)/np.pi*error))
    return 1-np.sqrt(k)/np.pi*integral


def bs_call(t, k, sigma):
    t, k, sigma = np.broadcast_arrays(np.asarray(t,float),np.asarray(k,float),np.asarray(sigma,float))
    w=np.maximum(t,0)*sigma*sigma
    root=np.sqrt(np.maximum(w,1e-30)); d=-np.log(k)/root+root/2
    return np.where(w>0,ndtr(d)-k*ndtr(d-root),np.maximum(1-k,0))


def _inputs(t,k,state):
    t,k,state=np.broadcast_arrays(np.asarray(t,float),np.asarray(k,float),np.asarray(state,float))
    if not all(np.isfinite(a).all() for a in (t,k,state)) or np.any(k<=0) or np.any(state<=0):
        raise ValueError('finite inputs, positive strike and state required')
    return t,k,state


def _heston_cf(u,t,v,kappa,theta,xi,rho):
    b=kappa-rho*xi*1j*u
    d=np.sqrt(b*b+xi*xi*(u*u+1j*u))
    g=(b-d)/(b+d)
    e=np.exp(-d*t)
    c=kappa*theta/xi**2*((b-d)*t-2*np.log((1-g*e)/(1-g)))
    dd=(b-d)/xi**2*(1-e)/(1-g*e)
    return np.exp(c+dd*v)


def price_heston(t,k,v,kappa,theta,volvol,rho):
    """Heston call via Little-Trap CF and 256-point Gauss-Legendre inversion.

    Each quote integrates to max(200,min(2000,40/sqrt(expected variance))).
    No clipping hides quadrature or arbitrage failures.
    """
    t,k,v=_inputs(t,k,v)
    if not np.isfinite([kappa,theta,volvol,rho]).all() or kappa<=0 or theta<=0 or volvol<0 or abs(rho)>=1:
        raise ValueError('invalid Heston parameters')
    positive=np.maximum(t,0)
    integrated=theta*positive+(v-theta)*(-np.expm1(-kappa*positive))/kappa
    if volvol < 1e-5:
        return bs_call(positive,k,np.sqrt(np.maximum(integrated,0)/np.maximum(positive,1e-30)))
    # Flatten observations, retain vectorized quadrature axis.
    tf=positive.ravel()[:,None];vf=v.ravel()[:,None];kf=k.ravel()[:,None]
    upper=np.clip(40/np.sqrt(np.maximum(integrated.ravel(),1e-8)),200,2000)[:,None]
    u=(_NODES+1)*upper/2;weights=_WEIGHTS*upper/2
    phase=np.exp(-1j*u*np.log(kf))/(1j*u)
    f1=_heston_cf(u-1j,tf,vf,kappa,theta,volvol,rho)
    f2=_heston_cf(u,tf,vf,kappa,theta,volvol,rho)
    p1=.5+np.sum(weights*np.real(phase*f1),axis=1)/np.pi
    p2=.5+np.sum(weights*np.real(phase*f2),axis=1)/np.pi
    result=p1-k.ravel()*p2
    # Oscillatory phases can be catastrophically under-resolved at short time
    # and low variance. Detect before trusting bounds; bounds alone miss
    # convexity failures. Adaptive damped inversion handles those quotes.
    positive_mask=positive.ravel()>0
    fallback=positive_mask&((upper.ravel()*np.abs(np.log(k.ravel()))>60)|
             ~np.isfinite(result)|(result<np.maximum(1-k.ravel(),0)-1e-10)|(result>1+1e-10))
    _PRICING['quotes']+=int(np.count_nonzero(positive_mask))
    _PRICING['fallback_quotes']+=int(np.count_nonzero(fallback))
    for j in np.flatnonzero(fallback):
        result[j]=_lewis_quad(float(positive.flat[j]),float(k.flat[j]),float(v.flat[j]),kappa,theta,volvol,rho)
    result=result.reshape(t.shape)
    return np.where(t<=0,np.maximum(1-k,0),result)


def price_sabr(t,k,alpha,nu,rho):
    """Hagan lognormal beta=1 asymptotic call, NOT exact SABR expectation."""
    t,k,alpha=_inputs(t,k,alpha)
    if not np.isfinite([nu,rho]).all() or nu<0 or rho<=-1 or rho>0:
        raise ValueError('beta=1 SABR requires nu>=0 and -1<rho<=0')
    z=nu/alpha*(-np.log(k))
    root=np.sqrt(1-2*rho*z+z*z)
    x=np.log((root+z-rho)/(1-rho))
    ratio=np.divide(z,x,out=np.ones_like(z),where=np.abs(z)>1e-6)
    ratio=np.where(np.abs(z)<=1e-6,1-rho*z/2+(2-3*rho*rho)*z*z/12,ratio)
    correction=1+(rho*nu*alpha/4+(2-3*rho*rho)*nu*nu/24)*np.maximum(t,0)
    if np.any(correction<=0):
        raise ValueError('Hagan expansion gives nonpositive volatility')
    return bs_call(t,k,alpha*ratio*correction)


def price_model(kind,model,t,k,state):
    if kind=='heston':
        return price_heston(t,k,state,model['kappa'],model['theta'],model['volvol'],model['rho'])
    if kind=='sabr':
        return price_sabr(t,k,state,model['nu'],model['rho'])
    raise ValueError('kind must be heston or sabr')


def _quotes(t,k,prices):
    t,k,prices=np.broadcast_arrays(np.asarray(t,float),np.asarray(k,float),np.asarray(prices,float))
    if not t.size or not all(np.isfinite(a).all() for a in (t,k,prices)) or np.any(t<=0) or np.any(k<=0):
        raise ValueError('positive finite times and strikes required')
    if np.any(prices<np.maximum(1-k,0)-1e-10) or np.any(prices>=1):
        raise ValueError('normalized prices outside call bounds')
    return t,k,prices


def check_q_grid(kind,model,state,times=(.02,.1,.3,1.)):
    """Finite strike/calendar checks at the supplied current latent state."""
    t=np.unique(np.r_[0.,np.asarray(times,float)])
    k=np.linspace(.5,1.5,161)
    try:
        c=price_model(kind,model,t[:,None],k[None,:],state)
    except ValueError as error:
        return {'valid':False,'error':str(error),'scope':'finite_grid_only',
                'time_range':[float(t.min()),float(t.max())],'strike_range':[.5,1.5]}
    slopes=np.diff(c,axis=1)/np.diff(k)
    checks={'call_lower':float(np.min(c-np.maximum(1-k,0))), 'call_upper':float(np.min(1-c)),
            'strike_monotonic':float(np.min(-slopes)), 'slope_lower':float(np.min(slopes+1)),
            'strike_convex':float(np.min(np.diff(slopes,axis=1))),
            'calendar':float(np.min(np.diff(c,axis=0)))}
    return {'valid':bool(np.isfinite(c).all() and min(checks.values())>=-2e-6),
            'margins':checks,'scope':'finite_grid_only','time_range':[float(t.min()),float(t.max())],
            'strike_range':[.5,1.5],'shape':[len(t),len(k)]}


def fit_global(kind,t,k,prices,states,seed=20261112):
    """Fit global Q parameters with caller-supplied, already causal states.

    Equal-weight price MSE and five deterministic starts. This function never
    refits states using future observations and does not establish causality of
    a state array produced elsewhere.
    """
    tic=time.perf_counter();t,k,prices=_quotes(t,k,prices)
    t,k,prices,states=np.broadcast_arrays(t,k,prices,np.asarray(states,float))
    _inputs(t,k,states)
    if kind=='heston':
        names=['kappa','theta','volvol','rho'];bounds=[(.05,15.),(.0025,1.),(.02,2.),(-.95,.5)]
        initial=np.array([2.,np.clip(np.median(states),.0025,1.),.3,-.5])
    elif kind=='sabr':
        names=['nu','rho'];bounds=[(0.,2.),(-.95,0.)];initial=np.array([.3,-.4])
    else:
        raise ValueError('kind must be heston or sabr')
    def unpack(z):
        return dict(zip(names,map(float,z)))
    def objective(z):
        try:
            residual=price_model(kind,unpack(z),t,k,states)-prices
            return float(np.mean(residual*residual)) if np.isfinite(residual).all() else 1e6
        except ValueError:
            return 1e6
    rng=np.random.default_rng(seed);results=[]
    for j in range(5):
        z=initial if j==0 else np.array([rng.uniform(lo,hi) for lo,hi in bounds])
        r=minimize(lambda z:1e4*objective(z),z,method='L-BFGS-B',bounds=bounds,
                   options={'maxiter':200,'ftol':1e-12,'gtol':1e-7})
        results.append(r)
    attempts=[];candidates=[]
    grid_states=np.unique(np.quantile(states,[0,.5,1]))
    grid_times=np.unique(np.r_[.02,.1,.3,1.,t.ravel()])
    for j,r in enumerate(results):
        model=unpack(r.x)
        # Check every optimizer candidate at fixed state quantiles, never
        # compare calls with different historical states across maturities.
        grid=[check_q_grid(kind,model,float(state),grid_times) for state in grid_states]
        loss=objective(r.x);grid_valid=all(g['valid'] for g in grid)
        qualified=bool(r.success and np.isfinite(r.fun) and np.isfinite(loss) and grid_valid)
        attempts.append(dict(success=bool(r.success),qualified=qualified,grid_valid=bool(grid_valid),
            objective=loss,parameters=unpack(r.x),grid=grid,message=str(r.message),
            optimizer_status=int(r.status),iterations=int(r.nit)))
        candidates.append((not qualified,loss,r,model,grid,j))
    rejected,loss,r,model,grid,selected=min(candidates,key=lambda item:item[:2])
    model.update(kind=kind)
    model['diagnostics']={'valid':not rejected,
                          'objective':loss,'optimizer_success':bool(r.success),
                          'message':str(r.message),'seconds':time.perf_counter()-tic,
                          'seed':int(seed),'starts':5,'grid':grid,'scope':'finite_grid_only',
                          'attempts':attempts,'qualified_candidates':sum(a['qualified'] for a in attempts),
                          'selected_start':int(selected),'grid_state_quantiles':grid_states.tolist(),
                          'selection_rule':'minimum_price_MSE_among_optimizer_success_and_all_Q_grids_valid_candidates;minimum_failed_MSE_if_none',
                          'pricing':'Heston_CF_Gauss256_adaptive_Lewis_fallback' if kind=='heston' else 'Hagan_beta1_asymptotic',
                          'quadrature':pricing_diagnostics() if kind=='heston' else None,
                          'objective_definition':'equal_weight_normalized_call_price_MSE'}
    return model


def filter_state(kind,model,t,k,prices,previous=None):
    """One current cross-section only: price-MSE latent state estimate.

    previous is only a candidate in the bounded one-dimensional search; no
    smoothing or future observations are used. Call sequentially in time.
    """
    t,k,prices=_quotes(t,k,prices)
    bounds=(.0004,1.) if kind=='heston' else (.02,2.)
    def loss(state):
        return float(np.mean((price_model(kind,model,t,k,state)-prices)**2))
    result=minimize_scalar(loss,bounds=bounds,method='bounded',options={'xatol':1e-10})
    if not result.success:
        raise RuntimeError('state filtering failed: '+str(result.message))
    state=float(result.x)
    if previous is not None and bounds[0]<=previous<=bounds[1] and loss(previous)<loss(state):
        state=float(previous)
    return state


def estimate_p_transitions(kind,states,next_states,dt,spots=None,next_spots=None):
    """Estimate P dynamics from explicit TRAINING transitions only.

    dt is years per transition. Caller is responsible for training cutoff and
    adjacency; transitions may not bridge held-out gaps. Q parameters are not
    reused as P drift estimates.
    """
    x,y,dt=np.broadcast_arrays(np.asarray(states,float),np.asarray(next_states,float),np.asarray(dt,float))
    x,y,dt=x.ravel(),y.ravel(),dt.ravel()
    if len(x)<3 or not np.isfinite([x,y,dt]).all() or np.any(x<=0) or np.any(y<=0) or np.any(dt<=0):
        raise ValueError('at least three finite positive-state, positive-dt training transitions required')
    if kind=='heston':
        scale=np.sqrt(x*dt)
        fit=least_squares(lambda p:(y-x-p[0]*(p[1]-x)*dt)/scale,
                          [2.,float(np.mean(x))],bounds=([1e-4,1e-6],[30.,4.]))
        kap,theta=fit.x;residual=(y-x-kap*(theta-x)*dt)/scale
        xi=float(np.sqrt(np.mean(residual**2)))
        result={'state_kappa':float(kap),'state_theta':float(theta)}
    elif kind=='sabr':
        r=np.log(y/x);log_drift=float(np.sum(r)/np.sum(dt))
        residual=(r-log_drift*dt)/np.sqrt(dt)
        xi=float(np.sqrt(np.mean(residual**2)))
        result={'state_log_drift':log_drift}
    else:
        raise ValueError('kind must be heston or sabr')
    result['spot_drift']=0.
    result['diagnostics']={'state_volvol_empirical':xi,'rho_empirical':None}
    if (spots is None)!=(next_spots is None):
        raise ValueError('both spot arrays are required')
    if spots is not None:
        s=np.broadcast_to(np.asarray(spots,float),x.shape);sn=np.broadcast_to(np.asarray(next_spots,float),x.shape)
        if not np.isfinite([s,sn]).all() or np.any(s<=0) or np.any(sn<=0):
            raise ValueError('positive finite spot transitions required')
        spot_var=x if kind=='heston' else x*x
        logs=np.log(sn/s);mu=float(np.sum(logs+.5*spot_var*dt)/np.sum(dt))
        spot_res=(logs-(mu-.5*spot_var)*dt)/np.sqrt(spot_var*dt)
        result['spot_drift']=mu
        result['diagnostics']['rho_empirical']=float(np.clip(np.corrcoef(spot_res,residual)[0,1],-.999,.999)) if np.std(residual)>0 and np.std(spot_res)>0 else 0.
    result.update(transitions=len(x),total_years=float(np.sum(dt)),measure='P',
                  estimator='Euler_WLS' if kind=='heston' else 'log_increment_MLE')
    return result


def simulate_p(kind,model,p_parameters,spot0,state0,dt,shocks):
    """P paths from supplied INDEPENDENT standard-normal shocks [...,steps,2].

    Last dimension is spot shock then independent factor shock. The same array
    can be shared between competing models. Output includes initial point.
    Q model rho and volvol/nu are frozen across measures; p_parameters supplies
    only state and spot drifts. Empirical diffusion diagnostics are ignored.
    """
    z=np.asarray(shocks,float)
    if z.ndim<2 or z.shape[-1]!=2 or not np.isfinite(z).all() or spot0<=0 or state0<=0:
        raise ValueError('positive initial values and finite [...,steps,2] shocks required')
    steps=z.shape[-2];dts=np.broadcast_to(np.asarray(dt,float),(steps,))
    if np.any(dts<=0) or not np.isfinite(dts).all():
        raise ValueError('positive finite dt required')
    rho=model['rho']
    xi=model['volvol'] if kind=='heston' else model['nu']
    if abs(rho)>=1:
        raise ValueError('correlation must be inside (-1,1)')
    shape=z.shape[:-2]+(steps+1,)
    spot=np.empty(shape);state=np.empty(shape);spot[...,0]=spot0;state[...,0]=state0
    for j,h in enumerate(dts):
        a=np.maximum(state[...,j],0);zs=z[...,j,0];za=rho*zs+np.sqrt(1-rho*rho)*z[...,j,1]
        sigma=np.sqrt(a) if kind=='heston' else a
        spot[...,j+1]=spot[...,j]*np.exp((p_parameters['spot_drift']-.5*sigma*sigma)*h+sigma*np.sqrt(h)*zs)
        if kind=='heston':
            state[...,j+1]=np.maximum(a+p_parameters['state_kappa']*(p_parameters['state_theta']-a)*h+xi*np.sqrt(a*h)*za,0)
        elif kind=='sabr':
            # New estimates explicitly store log drift. Legacy callers may
            # supply arithmetic alpha drift under the frozen Q diffusion.
            drift=p_parameters.get('state_log_drift')
            if drift is None:
                drift=p_parameters['state_drift']-.5*xi*xi
            state[...,j+1]=a*np.exp(drift*h+xi*np.sqrt(h)*za)
        else:
            raise ValueError('kind must be heston or sabr')
    return {'spot':spot,'state':state}


def q_derivatives(kind,model,t,k,state):
    """At S=1: price, Delta, Gamma, C_state, C_statestate, C_Sstate, LMV.

    LMV local hedge = Delta + rho * xi * C_state (Heston) or
    Delta + rho * nu * C_state (beta=1 SABR). Diffusion parameters are shared
    across P/Q; this instantaneous LMV does not depend on either drift.
    """
    if state<=0:
        raise ValueError('positive state required for central differences')
    hs=1e-4;hy=min(state*.01,max(1e-5,state*1e-3))
    def call(s,y):
        return float(s*price_model(kind,model,t,k/s,y))
    c=call(1,state);sp=call(1+hs,state);sm=call(1-hs,state)
    yp=call(1,state+hy);ym=call(1,state-hy)
    delta=(sp-sm)/(2*hs);state_d=(yp-ym)/(2*hy)
    cross=(call(1+hs,state+hy)-call(1+hs,state-hy)-call(1-hs,state+hy)+call(1-hs,state-hy))/(4*hs*hy)
    xi=model['volvol'] if kind=='heston' else model['nu']
    return {'price':c,'delta':delta,'gamma':(sp-2*c+sm)/hs**2,
            'state_derivative':state_d,'state_second':(yp-2*c+ym)/hy**2,
            'spot_state_cross':cross,'lmv':delta+model['rho']*xi*state_d}
