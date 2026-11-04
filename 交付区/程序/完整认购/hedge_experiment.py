import copy
import csv
from concurrent.futures import ProcessPoolExecutor
from itertools import groupby
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from contract_ledger import ledger

GROUPS = ("GBM DH", "Neural SDE DH", "Neural SDE DH factors")

def read_paths(source, end_years):
    result = {year:[] for year in end_years}
    with source.open(encoding="utf8",newline="") as f:
        for _, rows in groupby(csv.DictReader(f),lambda r:r["episode"]):
            path = list(rows)
            year = int(path[0]["split_end"])
            if year in result:
                result[year].append(path)
    return [result[year] for year in end_years]

def features(spot, strike, tau, rv, xi=None):
    base = np.stack([np.log(spot/strike)*20,tau*12,rv*4],axis=-1)
    if xi is not None:
        base = np.concatenate([base,xi],axis=-1)
    return base.astype(np.float32)

def real_inputs(paths, sim, with_xi):
    xi = dict(zip(sim["real_dates"],sim["real_xi"]))
    returns = sim["returns"]
    rolling = np.sqrt(252*np.mean(np.lib.stride_tricks.sliding_window_view(returns,20)**2,axis=-1))
    rv = dict(zip(sim["return_dates"][19:],rolling))
    horizon = max(map(len,paths))-1
    values = np.zeros((len(paths),horizon,5 if with_xi else 3),dtype=np.float32)
    units = np.ones((len(paths),horizon),dtype=np.float32)
    for i,path in enumerate(paths):
        dates = [r["date"] for r in path[:-1]]
        close = np.array([float(r["spot"]) for r in path[:-1]])
        strike = np.array([float(r["strike"]) for r in path[:-1]])
        tau = (np.datetime64(path[-1]["date"])-np.array(dates,dtype="datetime64[D]")).astype(float)/365
        values[i,:len(dates)] = features(close,strike,tau,np.array([rv[d] for d in dates]),
                                       np.array([xi[d] for d in dates]) if with_xi else None)
        units[i,:len(dates)] = [int(r["unit"]) for r in path[:-1]]
    assert np.isfinite(values).all()
    return torch.from_numpy(values),torch.from_numpy(units)

def evaluate(model, paths, inputs, fee, output=None, label=None):
    f, units = inputs
    shares = torch.zeros(len(paths))
    ratios = []
    with torch.no_grad():
        first = F.linear(f,model[0].weight[:,:-1],model[0].bias)
        for t in range(f.shape[1]):
            prev = shares/units[:,t]
            ratio = model[1:](first[:,t]+prev[:,None]*model[0].weight[:,-1]).squeeze(-1)
            shares = ratio*units[:,t]
            ratios.append(ratio.numpy())
    ratios = np.array(ratios).T
    errors = []
    for i,path in enumerate(paths):
        notional = int(path[0]["unit"])*float(path[0]["spot"])
        book = ledger(path,ratios[i,:len(path)-1],float(path[0]["option_close"]),fee)
        errors.append(book[-1]["wealth"]/notional)
        if output is not None:
            output.writerows(dict(**label,episode=path[0]["episode"],expiry=path[0]["expiry"],
                                  initial_notional=notional,**row) for row in book)
    return float(np.sqrt(np.mean(np.square(errors))))

def train_one(data_dir, cfg, name, bp, seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    data = Path(data_dir)
    tag = f"{name.replace(' ','_')}_{bp}_{seed}"
    job = data/"hedge_jobs"
    status = job/f"{tag}.json"
    weights = job/f"{tag}.pt"
    book_file = job/f"{tag}.csv"
    if status.exists() and weights.exists() and book_file.exists():
        return json.loads(status.read_text(encoding="utf8"))
    sim = np.load(data/"joint_paths.npz")
    with_xi = name.endswith("factors")
    train_n = cfg["simulation_train"]
    time = sim["time"]
    assert len(time)==train_n+cfg["simulation_valid"]
    spot = sim["gbm" if name=="GBM DH" else "spot"]
    past = np.concatenate([sim["warm"],np.diff(np.log(spot),axis=1)],axis=1)
    rv = np.sqrt(252*np.mean(np.lib.stride_tricks.sliding_window_view(past,20,axis=1)**2,axis=-1))
    strike = sim["strike"]
    x = torch.from_numpy(features(spot,strike[:,None],time[:,-1:]-time,rv,
                                  sim["xi"] if with_xi else None)[:,:-1].copy())
    prices = torch.from_numpy((spot/spot[:,:1]).astype(np.float32))
    active = torch.from_numpy(np.diff(time,axis=1)>0)
    terminal = torch.from_numpy(((sim["premium"]-np.maximum(spot[:,-1]-strike,0))/spot[:,0]).astype(np.float32))
    assert torch.isfinite(x).all()
    validation,test = read_paths(data/"episode_days.csv",cfg["split_ends"][1:])
    real_val = real_inputs(validation,sim,with_xi)
    model = nn.Sequential(nn.Linear(x.shape[-1]+1,32),nn.Tanh(),nn.Linear(32,32),
                          nn.Tanh(),nn.Linear(32,1),nn.Sigmoid())
    opt = torch.optim.Adam(model.parameters(),lr=.003)
    fee = bp/10000

    def objective(idx):
        f,p,mask = x[idx],prices[idx],active[idx]
        pnl = terminal[idx]
        prev = torch.zeros_like(pnl)
        # Unbind once so autograd accumulates the time-axis gradient in one pass.
        first = F.linear(f.transpose(0,1).contiguous(),model[0].weight[:,:-1],model[0].bias)
        prev_weight = model[0].weight[:,-1]
        rest = model[1:]
        for step, latent in enumerate(first.unbind(dim=0)):
            target = rest(latent+prev[:,None]*prev_weight).squeeze(-1)
            target = torch.where(mask[:,step],target,prev)
            pnl = pnl+target*(p[:,step+1]-p[:,step])-fee*(target-prev).abs()*p[:,step]
            prev = target
        return (pnl-fee*prev*p[:,-1]).square().mean()*10000

    best, stale, trace = float("inf"),0,[]
    for epoch in range(1,cfg["epochs"]+1):
        model.train()
        for indices in torch.randperm(train_n).split(cfg["batch_size"]):
            opt.zero_grad()
            objective(indices).backward()
            opt.step()
        if epoch%cfg["validation_every"]:
            continue
        model.eval()
        with torch.no_grad():
            sim_rmse = np.sqrt(objective(slice(train_n,None)).item())/100
        val = evaluate(model,validation,real_val,fee)
        assert np.isfinite(val) and np.isfinite(sim_rmse)
        trace.append(dict(epoch=epoch,validation_rmse=val,simulation_rmse=sim_rmse))
        if val<best:
            best,best_epoch,stale = val,epoch,0
            checkpoint = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        print(name,bp,seed,epoch,round(val,7),flush=True)
        if stale>=cfg["patience"]:
            break
    torch.save(checkpoint,weights)
    model.load_state_dict(torch.load(weights,map_location="cpu",weights_only=True))
    assert abs(evaluate(model,validation,real_val,fee)-best)<1e-10
    label = dict(group=name,bp=bp,seed=seed)
    fields = [*label,"episode","expiry","initial_notional","date","shares","traded","cost",
              "accrued","paid","receivable","liability","cash","wealth"]
    with book_file.open("w",encoding="utf8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        test_rmse = evaluate(model,test,real_inputs(test,sim,with_xi),fee,writer,label)
    result = dict(**label,best_epoch=best_epoch,epochs=epoch,validation_rmse=best,test_rmse=test_rmse,
                  book=book_file.name,trace=trace)
    status.write_text(json.dumps(result),encoding="utf8")
    return result

def main(data_dir, result_dir, config):
    data,result = Path(data_dir),Path(result_dir)
    cfg=json.loads(Path(config).read_text(encoding="utf8"))
    job=data/"hedge_jobs"
    job.mkdir(exist_ok=True)
    tasks = [(str(data),cfg,name,bp,seed) for bp in cfg["costs_bp"] for name in GROUPS for seed in cfg["seeds"]]
    with ProcessPoolExecutor(max_workers=6) as pool:
        runs = [future.result() for future in [pool.submit(train_one,*args) for args in tasks]]
    with (data/"dh_ledger.csv").open("w",encoding="utf8",newline="") as f:
        for i,run in enumerate(runs):
            with (job/run["book"]).open(encoding="utf8",newline="") as part:
                if i:
                    next(part)
                for row in part:
                    f.write(row)
    fields=["group","bp","seed","best_epoch","epochs","validation_rmse","test_rmse"]
    with (result/"training.csv").open("w",encoding="utf8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        writer.writeheader()
        writer.writerows(runs)
    (result/"training_trace.json").write_text(json.dumps(runs,indent=2),encoding="utf8")

if __name__ == "__main__":
    main(*sys.argv[1:])
