"""Train the three static-constraint-preserving price representations."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from bs_baseline import delta
from contract_ledger import ledger


def grid_mix(tau,k,ts,ks):
    tau,k = np.clip(tau,ts[0],ts[-1]),np.clip(k,ks[0],ks[-1])
    i = np.clip(np.searchsorted(ts,tau)-1,0,len(ts)-2)
    j = np.clip(np.searchsorted(ks,k)-1,0,len(ks)-2)
    x = (tau-ts[i])/(ts[i+1]-ts[i])
    y = (k-ks[j])/(ks[j+1]-ks[j])
    out = np.zeros((len(tau),len(ts),len(ks)))
    for di,wi in ((0,1-x),(1,x)):
        for dj,wj in ((0,1-y),(1,y)):
            out[np.arange(len(tau)),i+di,j+dj] = wi*wj
    return out.reshape(len(tau),-1)


def bs_ratios(path):
    iv = .25
    ratios = []
    for row in path[:-1]:
        ratio,iv,_ = delta(row,iv)
        ratios.append(ratio)
    return ratios


def weights(data,episodes,end):
    ts,ks = np.unique(data["grid_t"]),np.unique(data["grid_k"])
    coverage, errors, outside = [], [], []
    for _,p in episodes[episodes.split_end==end].groupby("episode"):
        path = p.to_dict("records")
        bs = bs_ratios(path)
        book = ledger(path,bs,float(path[0]["option_close"]),.0005)
        errors.append(book[-1]["wealth"]/(float(path[0]["spot"])*int(path[0]["unit"])))
        tau = (np.datetime64(path[0]["expiry"])-p.date.to_numpy(dtype="datetime64[D]")).astype(float)[:-1]/365
        k = (p.strike/p.spot).to_numpy()[:-1]
        coverage.append(np.mean(grid_mix(tau,k,ts,ks)**2,axis=0))
        outside.append(np.mean((tau<ts[0])|(tau>ts[-1])|(k<ks[0])|(k>ks[-1])))
    errors = np.array(errors)
    scale = max(np.median(np.abs(errors)),1e-6)
    hardness = np.clip(1+(errors/scale)**2,1,4)
    result = [np.ones(len(data["grid_t"]))]
    for h in [np.ones(len(errors)),hardness]:
        q = np.mean(np.asarray(coverage)*h[:,None],axis=0)
        u = np.sqrt(q+.01*q.mean())
        u = np.clip(u/u.mean(),.5,2)
        result.append(u/u.mean())
    return result, float(np.mean(outside))


def decode(data,train,w):
    from cvxopt import matrix,solvers
    from marketmodel.factors import DecodeFactor
    from scipy.optimize import linprog
    solvers.options["show_progress"] = False
    np.random.seed(2026)
    a,b,calls = (data[k] for k in ("A","b","calls"))
    d = np.sqrt(w)
    y = calls*d
    center = y[train].mean(axis=0)
    residual = y[train]-center
    pca,score = DecodeFactor.decode_pca_factor(residual,1)
    ac = a/d
    rhs = (b-ac@center)[:,None] - (ac@pca.T)@score.T
    sa,_,_ = DecodeFactor.decode_stcarb_factor((residual-score@pca).copy(),1,4,ac,rhs.copy())
    basis = np.vstack([pca,sa])
    H = basis@basis.T
    assert np.linalg.cond(H)<1e8
    W,bounds = ac@basis.T,b-ac@center
    norms = np.linalg.norm(W,axis=1)
    lp = linprog([0,0,-1], A_ub=np.c_[-W,norms], b_ub=-bounds,
                 bounds=[(None,None)]*3,method="highs")
    assert lp.success and lp.x[2]>1e-8
    states = (y-center)@np.linalg.pinv(basis)
    infeasible = np.any(states@W.T-bounds<-1e-9,axis=1)
    for i in np.flatnonzero(infeasible):
        fit = solvers.qp(matrix(H),matrix(-basis@(y[i]-center)),matrix(-W),matrix(-bounds))
        assert fit["status"]=="optimal"
        states[i] = np.asarray(fit["x"]).ravel()
    assert np.min(states@W.T-bounds)>=-1e-7
    error = ((center+states@basis)-y)/d
    return dict(xi=states,center=center/d,basis=basis/d,W=W,bounds=bounds,
                weights=w,weighted_rmse=np.sqrt(np.mean(error**2*w,axis=1)),
                reconstruction_rmse=np.sqrt(np.mean(error**2,axis=1)),
                projected=infeasible,condition=np.linalg.cond(H),radius=lp.x[2])


def run(surface,episode_file,upstream,out,config):
    sys.path.insert(0,str(upstream))
    cfg=json.loads(Path(config).read_text(encoding="utf-8-sig"))
    data=dict(np.load(surface))
    episodes=pd.read_csv(episode_file)
    end=cfg["split_ends"][0]
    train=data["dates"]<str(end+1)
    all_weights,outside=weights(data,episodes,end)
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    for i,w in enumerate(all_weights):
        shrunk=False
        try:
            fitted=decode(data,train,w)
        except (AssertionError, ValueError):
            if i==0:
                raise
            fitted=decode(data,train,.5+.5*w)
            shrunk=True
        np.savez(out/f"C{i}.npz",dates=data["dates"],spot=data["spot"],grid_t=data["grid_t"],
                 grid_k=data["grid_k"],outside=outside,shrunk=shrunk,**fitted)
        print("C"+str(i),"condition",fitted["condition"],"shrink",shrunk,flush=True)


if __name__=="__main__":
    run(*sys.argv[1:])

