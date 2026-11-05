"""Fit joint NSDE members and export portable simulation arrays."""
import inspect
import json
from pathlib import Path
import sys
import textwrap

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection
import tensorflow as tf
import tensorflow_probability as tfp

from contract_ledger import EVENTS


def fit_member(data, cal, cfg, out, seed, member):
    from marketmodel.neuralsde import Loss, Train
    from marketmodel.factors import PrepTrainData

    boundary = cfg["boundary"]
    tf.keras.utils.set_random_seed(seed)
    dates = data["dates"]
    train = dates < str(cfg["split_ends"][0]+1)
    xi_scale = data["xi"][train].std(axis=0)
    xi = data["xi"]/xi_scale
    A = data["W"]*xi_scale
    norms = np.linalg.norm(A, axis=1)
    keep = norms > 1e-10
    W = A[keep]/norms[keep,None]
    b = data["bounds"][keep]/norms[keep]
    center = linprog([0,0,-1], A_ub=np.c_[-W,np.ones(len(W))], b_ub=-b,
                     bounds=[(None,None)]*3, method="highs")
    assert center.success and center.x[2] > 0
    anchor, radius = center.x[:2], center.x[2]
    vertices = HalfspaceIntersection(np.c_[-W,b],anchor).intersections
    faces = np.flatnonzero(np.sum(np.abs(vertices@W.T-b)<1e-8,axis=0)>=2)
    unique = np.unique(np.round(np.c_[W[faces], b[faces]], 9),axis=0,return_index=True)[1]
    W, b = W[faces[unique]], b[faces[unique]]
    Wbar = np.c_[np.zeros(len(W)),W]
    rho, eps = radius*.1, 1.0
    # A face normal can miss the shrunken polygon in a narrow corner.
    # Shrink the numerical boundary band without moving any price constraint.
    for _ in range(9):
        try:
            interior, _, _ = PrepTrainData.calc_drift_correction(W,b,xi[:1],rho,eps)
            break
        except ValueError as exc:
            if "greatest lower bound" not in str(exc):
                raise
            rho *= .5
    else:
        raise ValueError("No feasible drift-correction band after eight halvings")
    assert np.min(interior @ W.T - b) >= -1e-9
    print(f"Drift correction band/radius: {rho/radius:.8g}",flush=True)
    # Move observations infinitesimally inside so the likelihood is nonsingular.
    xi = .999*xi + .001*anchor
    idx = cal.index.get_indexer(dates)
    dt = np.diff(dates.astype("datetime64[D]")).astype(float)/365
    valid = train[1:] & (np.diff(idx)==1) & ~np.isin(dates[1:], [x[0] for x in EVENTS])
    log_s = np.log(data["spot"])
    dx = np.diff(np.c_[log_s,xi],axis=0)[valid]
    dtime = dt[valid]
    n = int(.9 * len(dx))
    fit = np.arange(n)
    if member:
        original = np.flatnonzero(valid)[:n]
        starts = np.flatnonzero(original[19:] - original[:-19] == 19)
        if not len(starts):
            raise ValueError("No contiguous 20-transition blocks")
        rng = np.random.default_rng(seed)
        fit = (rng.choice(starts, (n+19)//20)[:, None] + np.arange(20)).ravel()[:n]
    check = np.arange(n, len(dx))
    assert not np.intersect1d(fit, check).size
    spot_mu = dx[fit,0].sum()/dtime[fit].sum()
    spot_sigma = np.sqrt(np.mean((dx[fit,0]-spot_mu*dtime[fit])**2/dtime[fit]))
    multiplier = 10/radius

    def operators(x):
        distance = x@W.T-b
        nearest = np.argsort(distance,axis=1)
        first = nearest[:,0]
        direction = W[first]
        perp = np.c_[-direction[:,1], direction[:,0]]
        independent = np.abs(np.einsum("bi,bki->bk",perp,W[nearest]))>1e-8
        second = nearest[np.arange(len(x)),np.argmax(independent,axis=1)]
        scales = np.sqrt(PrepTrainData.normalise_dist_diffusion(
            np.maximum(distance[np.arange(len(x))[:,None], np.c_[first,second]],0),multiplier,1))
        omega = np.zeros((len(x),3,3))
        omega[:,0,0] = spot_sigma
        omega[:,1,1:] = direction*scales[:,0,None]
        omega[:,2,1:] = perp*scales[:,1,None]
        dirs = interior[None,:,:]-x[:,None,:]
        tol = PrepTrainData.normalise_dist_drift(np.minimum(np.maximum(distance,0),30*rho),rho,eps)
        return omega, dirs, tol

    x = xi[:-1][valid]
    omega, dirs, tol = operators(x)
    projected = np.linalg.solve(omega.transpose(0,2,1),dx[...,None])[...,0]
    y = np.column_stack([projected,omega.reshape(-1,9),np.abs(np.linalg.det(omega)),
                         dirs.reshape(len(x),-1),tol,dtime,np.full(len(x),spot_mu)])
    # Use the author's joint likelihood, adapting its scalar step to row-specific ACT/365.
    code = textwrap.dedent(inspect.getsource(Loss.loss_xiS)).replace("@staticmethod\n","",1)
    code = code.replace("1. / dt * tf.reduce_sum(tf.square(sol_dX))",
                        "tf.reduce_sum(tf.square(sol_dX) / y_true[:, -2:-1])")
    code = code.replace("dt * tf.reduce_sum(tf.square(sol_mu))",
                        "tf.reduce_sum(y_true[:, -2:-1] * tf.square(sol_mu))")
    scope = {"tf": tf, "tfp": tfp}
    exec(code,scope)
    nvar, mask = Train._identify_diagonal_entries(3)
    loss = scope["loss_xiS"](None,3,nvar,mask[:-1],W,Wbar)
    model = tf.keras.Sequential([tf.keras.layers.Input((2,)),
        tf.keras.layers.Dense(64,activation="tanh"), tf.keras.layers.Dense(64,activation="tanh"),
        tf.keras.layers.Dense(7)])
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4,clipnorm=10),loss=loss)
    history = model.fit(x[fit],y[fit],epochs=100,batch_size=128,
        validation_data=(x[check],y[check]),verbose=0,
        callbacks=[tf.keras.callbacks.TerminateOnNaN(),
                      tf.keras.callbacks.EarlyStopping(patience=12,restore_best_weights=True)])
    assert np.isfinite(history.history["loss"]).all()
    model.save_weights(str(out / f"market_{seed}.weights.h5"))
    model.load_weights(str(out / f"market_{seed}.weights.h5"))
    # NLL is evaluated directly in the common physical coordinates below.
    def coefficients(x):
        omega, dirs, tol = operators(x)
        pred = model(x.astype(np.float32)).numpy().astype(float)
        pred = np.c_[pred[:,:2],np.zeros(len(x)),pred[:,2:]]
        drift = np.c_[np.full(len(x),spot_mu),pred[:,6:]]
        gamma = np.maximum(-drift@Wbar.T-tol,0)/np.einsum("bki,ki->bk",dirs,W)
        drift[:,1:] += np.einsum("bk,bki->bi",gamma,dirs)
        tri = pred[:,:6].copy()
        diagonal = np.flatnonzero(np.ravel(mask))
        tri[:,diagonal] = np.exp(tri[:,diagonal])
        diffusion = (Train._fill_triu(tri,3)@omega).transpose(0,2,1)
        return drift,diffusion


    mu, diffusion = coefficients(x[check])
    covariance = diffusion @ diffusion.transpose(0, 2, 1) * dtime[check,None,None]
    residual = dx[check] - mu * dtime[check,None]
    sign, logdet = np.linalg.slogdet(covariance)
    nll = .5 * (logdet + np.sum(residual * np.linalg.solve(
        covariance,residual[...,None])[...,0],axis=1))
    assert (sign == 1).all() and np.isfinite(nll).all()
    return coefficients, nll, (xi, xi_scale, W, b, anchor, radius, spot_mu, spot_sigma)


def simulate(coefficients, start, t, shocks, geometry, subdivisions, boundary):
    _, _, W, b, anchor, radius, _, _ = geometry
    steps = np.diff(t,axis=1)
    simulated = np.empty((*t.shape,3))
    simulated[:,0] = start
    horizon = t.shape[1]
    subdivisions_list = (subdivisions,)
    diagnostics = []
    for subdivisions in subdivisions_list:
        state = start.copy()
        breaches = total = drift_only = closest_hit = 0
        days_hit = correction = max_correction = 0
        near_start = breached_near = 0
        last_corrected = np.zeros(len(start), dtype=bool)
        after_correction = exposed_after_correction = 0
        for i in range(horizon-1):
            active = steps[:,i]>0
            dt_i = steps[active,i,None]/subdivisions
            noise = shocks[active,i]
            any_hit = np.zeros(sum(active),dtype=bool)
            for j in range(subdivisions):
                z = noise[:,j] if subdivisions>1 else noise.sum(axis=1)/np.sqrt(2)
                x = state[active]
                drift,vol = coefficients(x[:,1:])
                start_slack = x[:,1:]@W.T-b
                drift_state = x + drift*dt_i
                trial = drift_state + np.einsum("bij,bj->bi",vol,z)*np.sqrt(dt_i)
                slack = trial[:,1:]@W.T-b
                violated = np.any(slack<0,axis=1)
                any_hit |= violated
                breaches += int(violated.sum())
                total += len(trial)
                drift_only += int(np.any(drift_state[:,1:]@W.T-b<0,axis=1).sum())
                closest_hit += int(np.sum(slack[np.arange(len(x)),start_slack.argmin(axis=1)]<0))
                near = start_slack.min(axis=1)<.01*radius
                near_start += int(near.sum())
                breached_near += int((near & violated).sum())
                after_correction += int((last_corrected[active] & violated).sum())
                exposed_after_correction += int(last_corrected[active].sum())
                last_corrected[active] = violated
                raw_factors = trial[:,1:].copy()
                if boundary=="reflect":
                    # Reflect the most violated unit-normal face, then check corners.
                    for _ in range(3):
                        face = slack.argmin(axis=1)
                        depth = slack[np.arange(len(trial)),face]
                        hit = depth < 0
                        if not hit.any():
                            break
                        trial[hit,1:] -= 2*depth[hit,None]*W[face[hit]]
                        slack = trial[:,1:]@W.T-b
                fallback = np.any(slack<0,axis=1)
                q = np.maximum(-slack,0)
                blend = np.max(q/(q+anchor@W.T-b),axis=1)
                trial[:,1:] += np.minimum(1,blend+fallback*1e-6)[:,None]*(anchor-trial[:,1:])
                lengths = np.linalg.norm(trial[:,1:]-raw_factors,axis=1)
                correction += lengths.sum()
                max_correction = max(max_correction,lengths.max(initial=0))
                assert np.min(trial[:,1:]@W.T-b)>=-1e-8
                state[active] = trial
            days_hit += int(any_hit.sum())
            if subdivisions==subdivisions_list[-1]:
                simulated[:,i+1] = state
        record = dict(boundary=boundary,subdivisions=subdivisions,breaches=breaches,steps=total,
                      drift_only=drift_only,closest_hit=closest_hit,days_hit=days_hit,
                      days=int(np.count_nonzero(steps)),near_start=near_start,breached_near=breached_near,
                      exposed_after_correction=exposed_after_correction,after_correction=after_correction,
                      correction_per_day=correction/np.count_nonzero(steps),max_correction=max_correction)
        diagnostics.append(record)
        print(record,flush=True)

    return simulated, diagnostics[0]


def reliability(states, times, coefficients, geometry, initial, chunk=32768):
    _, _, W, b, _, radius, _, _ = geometry
    n, h, _ = states.shape
    q = np.zeros((n,h,3), dtype=np.float32)
    q[:,0,0] = initial
    q[:,0,2] = 1
    q[:,:,1] = np.min(states @ W.T - b, axis=-1) / radius
    for left in range(0,n,max(1,chunk//h)):
        s = states[left:left+max(1,chunk//h)]
        dt = np.diff(times[left:left+len(s)],axis=1)
        active = dt > 0
        mu, vol = coefficients(s[:,:-1][active])
        cov = (vol[:,1:] @ vol[:,1:].transpose(0,2,1)) * dt[active,None,None]
        cov += 1e-6 * np.eye(2)
        res = np.diff(s,axis=1)[active] - mu[:,1:] * dt[active,None]
        residual = np.zeros_like(dt)
        residual[active] = np.sum(res*np.linalg.solve(cov,res[...,None])[...,0],axis=1)
        for j in range(1,h):
            q[left:left+len(s),j,0] = .94*q[left:left+len(s),j-1,0] + .06*residual[:,j-1]
    return q


def prepare_origins(episodes, dates, xi, cal, cfg):
    dates_to_xi = dict(zip(dates,xi))
    origins = [p for _,p in episodes.groupby("episode")
               if p.iloc[0].split_end == cfg["split_ends"][0]]
    dividends = pd.Series({x[0]:float(x[3]) for x in EVENTS}).reindex(cal.index,fill_value=0)
    returns = np.log((cal+dividends)/cal.shift()).to_numpy()
    rng = np.random.default_rng(cfg["origin_seed"])
    selected = rng.integers(len(origins),size=cfg["simulation_train"]+cfg["simulation_valid"])
    h = max(map(len,origins))
    t = np.zeros((len(selected),h))
    warm = np.empty((len(selected),20))
    start = np.empty((len(selected),3))
    strikes,premiums = np.empty(len(selected)),np.empty(len(selected))
    for i,j in enumerate(selected):
        p = origins[j]
        a = p.iloc[0]
        tt = (p.date.to_numpy(dtype="datetime64[D]")-np.datetime64(a.date)).astype(float)/365
        t[i,:len(tt)],t[i,len(tt):] = tt,tt[-1]
        k = cal.index.get_loc(a.date)
        warm[i] = returns[k-19:k+1]
        start[i] = [np.log(a.spot),*dates_to_xi[a.date]]
        strikes[i],premiums[i] = a.strike,a.option_close
    assert np.isfinite(warm).all()
    return t,warm,start,strikes,premiums,returns


def run(factors, spot_file, upstream, episode_file, out, config):
    sys.path.insert(0,str(upstream))
    tf.config.threading.set_intra_op_parallelism_threads(4)
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    cfg = json.loads(Path(config).read_text(encoding="utf-8-sig"))
    data = dict(np.load(factors))
    cal = pd.read_csv(spot_file,encoding="gb18030").set_index("交易日期")["收盘价(元)"].sort_index()
    episodes = pd.read_csv(episode_file)
    dates = data["dates"]
    fit0, nll0, geom = fit_member(data,cal,cfg,out,2026,0)
    xi, scale, W, b, anchor, radius, mu, sigma = geom
    t,warm,start,strike,premium,returns = prepare_origins(episodes,dates,xi,cal,cfg)
    count,h = t.shape
    dt = np.diff(t,axis=1)
    shocks4 = np.random.default_rng(cfg["shock_seed"]).normal(size=(count,h-1,4,3))
    shocks2 = shocks4.reshape(count,h-1,2,2,3).sum(axis=3)/np.sqrt(2)
    train = dates < str(cfg["split_ends"][0]+1)
    day = dates.astype("datetime64[D]").astype(float)/365
    # Estimate the initialization from actual legal training transitions.
    ix = cal.index.get_indexer(dates)
    valid = train[1:] & (np.diff(ix)==1) & ~np.isin(dates[1:],[x[0] for x in EVENTS])
    drift,vol = fit0(xi[:-1][valid])
    delta = np.diff(day)[valid]
    residual = np.diff(xi,axis=0)[valid] - drift[:,1:]*delta[:,None]
    cov = (vol[:,1:] @ vol[:,1:].transpose(0,2,1))*delta[:,None,None]+1e-6*np.eye(2)
    initial = float(np.median(np.sum(residual*np.linalg.solve(cov,residual[...,None])[...,0],axis=1)))
    date_to_i = dict(zip(dates,range(len(dates))))
    real_q = np.zeros((len(episodes),3),dtype=np.float32)
    for _, group in episodes.groupby("episode",sort=False):
        idx = [date_to_i[d] for d in group.date]
        real_q[group.index] = reliability(xi[idx][None],day[idx][None],fit0,geom,initial)[0]
    fit_rows = episodes.split_end.to_numpy() == cfg["split_ends"][0]
    lower, upper = np.quantile(real_q[fit_rows,:2],[.005,.995],axis=0)
    center = np.median(np.clip(real_q[fit_rows,:2],lower,upper),axis=0)
    spread = np.maximum(np.std(np.clip(real_q[fit_rows,:2],lower,upper),axis=0),1e-6)

    def normalize(q):
        q[...,:2] = (np.clip(q[...,:2],lower,upper)-center)/spread
        return q

    sim0, diag0 = simulate(fit0,start,t,shocks2,geom,2,"reflect")
    gbm = np.c_[start[:,0],start[:,0,None]+np.cumsum(
        mu*dt+sigma*np.sqrt(dt)*shocks2[:,:,:,0].sum(axis=2)/np.sqrt(2),axis=1)]
    diagnostics, accepted = [], []
    diagnostics.append(dict(seed=2026,accepted=True,nll=float(nll0.mean())))

    def checked(sim):
        slack = min((block @ (data["W"]*scale).T-data["bounds"]).min()
                    for block in np.array_split(sim[:,:,1:].reshape(-1,2),128))
        assert np.isfinite(sim).all() and slack >= -1e-7
        return float(slack)

    def diag_record(fit, geometry, sim, diag, seed):
        sample = slice(None,None,8)
        fine, fine_diag = simulate(fit,start[sample],t[sample],shocks4[sample],geometry,4,"reflect")
        coarse, coarse_diag = simulate(fit,start[sample],t[sample],shocks2[sample],geometry,2,"reflect")
        def moments(paths, clocks):
            increments = np.diff(paths,axis=1)[np.diff(clocks,axis=1)>0]
            return dict(quantiles=np.quantile(increments[:,0],[.01,.5,.99]).tolist(),
                        covariance=np.cov(increments.T).tolist())
        return dict(seed=seed,full=diag,paired2=coarse_diag,paired4=fine_diag,
                    moments2=moments(coarse,t[sample]),moments4=moments(fine,t[sample]),
                    full_slack=checked(sim),fine_slack=checked(fine))

    numeric = [diag_record(fit0,geom,sim0,diag0,2026)]
    for seed in range(2027,2031):
        try:
            fit,nll,g = fit_member(data,cal,cfg,out,seed,1)
            diff = nll-nll0
            blocks = np.array([diff[i:i+20].mean() for i in range(0,len(diff),20)])
            se = blocks.std(ddof=1)/np.sqrt(len(blocks))
            ok = bool(diff.mean() <= 2*se)
            record = dict(seed=seed,nll=float(nll.mean()),difference=float(diff.mean()),
                          se=float(se),accepted=ok)
            if ok:
                s,d = simulate(fit,start,t,shocks2,g,2,"reflect")
                numeric.append(diag_record(fit,g,s,d,seed))
                accepted.append(s)
            diagnostics.append(record)
            if len(accepted)==2:
                break
        except (ValueError,AssertionError,np.linalg.LinAlgError) as exc:
            diagnostics.append(dict(seed=seed,accepted=False,error=repr(exc)))

    real_mu, real_vol = fit0(xi)
    np.savez(out/"common.npz",time=t.astype("f"),warm=warm.astype("f"),strike=strike,
             premium=premium,real_dates=dates,real_xi=xi,return_dates=cal.index.to_numpy(dtype=str),
             returns=returns,real_q=normalize(real_q),gbm=np.exp(gbm).astype("f"),
             real_cov=real_vol @ real_vol.transpose(0,2,1),scale=scale)
    def save(name,s,ids):
        q = normalize(reliability(s[:,:,1:],t,fit0,geom,initial))
        np.savez(out/f"{name}.npz",spot=np.exp(s[:,:,0]).astype("f"),
                 xi=s[:,:,1:].astype("f"),quality=q,member=ids)
    save("single",sim0,np.zeros(count,dtype=np.int8))
    if accepted:
        m = len(accepted)+1
        n = cfg["simulation_train"]
        ids = np.r_[np.arange(n)%m,np.arange(count-n)%m].astype(np.int8)
        for j,s in enumerate(accepted,1):
            sim0[ids==j] = s[ids==j]
        save("pool",sim0,ids)
    (out/"market.json").write_text(json.dumps(dict(members=1+len(accepted),fit=diagnostics,
        numeric=numeric,initial=initial,lower=lower.tolist(),upper=upper.tolist(),
        center=center.tolist(),spread=spread.tolist()),allow_nan=False),encoding="utf8")


if __name__ == "__main__":
    run(*sys.argv[1:])




