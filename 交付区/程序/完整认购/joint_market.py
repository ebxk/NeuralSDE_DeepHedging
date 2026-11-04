import inspect
import json
from pathlib import Path
import sys
import textwrap

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection
import tensorflow as tf
import tensorflow_probability as tfp

from contract_ledger import EVENTS

def main(source, spot_file, upstream, out, config):
    import pandas as pd
    sys.path.insert(0, upstream)
    from marketmodel.neuralsde import Loss, Train
    from marketmodel.factors import PrepTrainData

    out = Path(out)
    cfg = json.loads(Path(config).read_text(encoding="utf8"))
    output = out
    boundary = cfg["boundary"]
    output.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(cfg["market_seed"])
    tf.config.threading.set_intra_op_parallelism_threads(4)
    data = np.load(source)
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
    interior, _, _ = PrepTrainData.calc_drift_correction(W,b,xi[:1],rho,eps)
    # Move observations infinitesimally inside so the likelihood is nonsingular.
    xi = .999*xi + .001*anchor
    cal = pd.read_csv(spot_file,encoding="gb18030").set_index("交易日期")["收盘价(元)"].sort_index()
    idx = cal.index.get_indexer(dates)
    dt = np.diff(dates.astype("datetime64[D]")).astype(float)/365
    valid = train[1:] & (np.diff(idx)==1) & ~np.isin(dates[1:], [x[0] for x in EVENTS])
    log_s = np.log(data["spot"])
    dx = np.diff(np.c_[log_s,xi],axis=0)[valid]
    dtime = dt[valid]
    spot_mu = dx[:,0].sum()/dtime.sum()
    spot_sigma = np.sqrt(np.mean((dx[:,0]-spot_mu*dtime)**2/dtime))
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
    if not (out/"joint.weights.h5").exists():
        model.compile(optimizer=tf.keras.optimizers.Adam(3e-4,clipnorm=10),loss=loss)
        history = model.fit(x,y,epochs=100,batch_size=128,validation_split=.1,verbose=0,
            callbacks=[tf.keras.callbacks.TerminateOnNaN(),tf.keras.callbacks.EarlyStopping(
                patience=12,restore_best_weights=True)])
        assert np.isfinite(history.history["loss"]).all()
        print("Joint likelihood:", len(x), len(history.epoch), min(history.history["val_loss"]),flush=True)
        model.save_weights(str(out/"joint.weights.h5"))
        prediction = model(x[:8]).numpy()
        model.load_weights(str(out/"joint.weights.h5"))
        np.testing.assert_allclose(prediction,model(x[:8]).numpy())
    else:
        model.load_weights(str(out/"joint.weights.h5"))

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

    episodes = pd.read_csv(out/"episode_days.csv")
    origins = [p for _,p in episodes.groupby("episode") if p.iloc[0].split_end==cfg["split_ends"][0]]
    date_to_xi = dict(zip(dates,xi))
    dividends = pd.Series({x[0]:float(x[3]) for x in EVENTS}).reindex(cal.index,fill_value=0)
    returns = np.log((cal+dividends)/cal.shift()).to_numpy()
    rng = np.random.default_rng(cfg["origin_seed"])
    selected = rng.integers(len(origins),size=cfg["simulation_train"]+cfg["simulation_valid"])
    horizon = max(len(p) for p in origins)
    t = np.zeros((len(selected),horizon))
    warm = np.empty((len(selected),20))
    start = np.empty((len(selected),3))
    strikes, premiums = np.empty(len(selected)),np.empty(len(selected))
    for i, j in enumerate(selected):
        p = origins[j]
        a = p.iloc[0]
        times = (p.date.to_numpy(dtype="datetime64[D]")-np.datetime64(a.date)).astype(float)/365
        t[i,:len(times)] = times
        t[i,len(times):] = times[-1]
        k = cal.index.get_loc(a.date)
        warm[i] = returns[k-19:k+1]
        start[i] = [np.log(a.spot),*date_to_xi[a.date]]
        strikes[i],premiums[i] = a.strike,a.option_close
    assert np.isfinite(warm).all()
    rng = np.random.default_rng(cfg["shock_seed"])
    shocks = rng.normal(size=(len(selected),horizon-1,2,3))
    steps = np.diff(t,axis=1)
    simulated = np.empty((len(selected),horizon,3))
    simulated[:,0] = start
    gbm = np.empty(t.shape)
    gbm[:,0] = start[:,0]
    subdivisions_list = (2,)
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
                max_correction = max(max_correction,lengths.max())
                assert np.min(trial[:,1:]@W.T-b)>=-1e-8
                state[active] = trial
            days_hit += int(any_hit.sum())
            if subdivisions==subdivisions_list[-1]:
                simulated[:,i+1] = state
                gbm[:,i+1] = gbm[:,i]+spot_mu*steps[:,i]+spot_sigma*np.sqrt(steps[:,i])*shocks[:,i,:,0].sum(axis=1)/np.sqrt(2)
        record = dict(boundary=boundary,subdivisions=subdivisions,breaches=breaches,steps=total,
                      drift_only=drift_only,closest_hit=closest_hit,days_hit=days_hit,
                      days=int(np.count_nonzero(steps)),near_start=near_start,breached_near=breached_near,
                      exposed_after_correction=exposed_after_correction,after_correction=after_correction,
                      correction_per_day=correction/np.count_nonzero(steps),max_correction=max_correction)
        diagnostics.append(record)
        print(record,flush=True)
    pd.DataFrame(diagnostics).to_csv(output/"boundary_diagnostics.csv",index=False)
    slack = min(((block*xi_scale)@data["W"].T-data["bounds"]).min()
                for block in np.array_split(simulated[:,:,1:].reshape(-1,2),128))
    assert np.isfinite(simulated).all() and slack>=-1e-7
    print("Full-grid minimum slack:",slack,flush=True)
    np.savez_compressed(output/"joint_paths.npz",time=t,spot=np.exp(simulated[:,:,0]),
                        xi=simulated[:,:,1:],gbm=np.exp(gbm),strike=strikes,premium=premiums,
                        warm=warm,real_dates=dates,real_xi=xi,return_dates=cal.index.to_numpy(dtype=str),
                        returns=returns, W=W, b=b, anchor=anchor, interior=interior,
                        xi_scale=xi_scale, spot_mu=spot_mu, spot_sigma=spot_sigma)

if __name__ == "__main__":
    main(*sys.argv[1:])
