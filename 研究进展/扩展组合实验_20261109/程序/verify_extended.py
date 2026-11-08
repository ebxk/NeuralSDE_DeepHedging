"""Executable checks for curve extensions, alignment, and legacy replay."""
import json
from collections import Counter
import csv

import numpy as np
import torch

from prepare_common import write_json
from contract_ledger import ledger
from hedge_experiment import real_inputs
from extended_policy import DensePolicy, load_legacy, make_policy, rollout, simulated_pnl
from extended_train import predictions
from extended_registry import Job
from extended_curve import bs_delta, curve
from extended_data import Inputs, batch
from extended_registry import registry


def verify_original_interface(project, cfg):
    """Load all original 32-wide models into the new account interface."""
    data = project/"工作区/数据/处理后"
    base, old = data/"创新组合实验"/"2025", data/"完整认购"
    inputs = Inputs(base, base/"C1", base/"C1.npz", cfg["folds"][0])
    paths, observed = inputs.observed(2025)
    state = np.load(old/"joint_paths.npz")
    expected = {}
    with (old/"dh_ledger.csv").open(encoding="utf8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["date"] == row["expiry"]:
                key = (row["group"], int(row["bp"]), int(row["seed"]), row["episode"])
                expected[key] = float(row["wealth"])/float(row["initial_notional"])
    worst, rounding, checked = 0., 0., 0
    for file in (old/"hedge_jobs").glob("*.pt"):
        status = file.with_suffix(".json")
        if not status.exists():
            continue
        meta = json.loads(status.read_text(encoding="utf8"))
        name, bp, seed = meta["group"], meta["bp"], meta["seed"]
        has = name.endswith("factors")
        f, _ = real_inputs(paths, state, has)
        model = DensePolicy("legacy", current=5 if has else 3, width=32)
        load_legacy(model, torch.load(file, weights_only=True))
        job = Job(2025,"C0",0,"off",False,False,bp,0,seed,"single","narrow","baseline")
        pnl, ratios, _ = predictions(model,job,dict(observed,f=f),size=len(paths))
        for p,r,new in zip(paths,ratios,pnl):
            numeraire = float(p[0]["spot"])*int(p[0]["unit"])
            old_pnl = float(ledger(p,r[:len(p)-1],float(p[0]["option_close"]),bp/10000)[-1]["wealth"])/numeraire
            worst = max(worst,abs(old_pnl-expected[name,bp,seed,p[0]["episode"]]))
            rounding = max(rounding,abs(new-old_pnl))
        checked += 1
    state.close()
    assert checked == 18 and worst <= 1e-8, (checked,worst)
    return worst, rounding


def verify(project, root, cfg):
    torch.set_num_threads(1)
    import unittest
    import verify_policy
    assert unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(verify_policy)).wasSuccessful()
    expected = dict(main=1152,narrow=144,teacher=72,history=48,delta=48,bridge=168,cost=144)
    assert Counter(x.family for x in registry(cfg).values()) == expected
    previous = project/"工作区/数据/处理后"
    original = previous/"创新组合实验"
    worst, rounding = verify_original_interface(project, cfg)
    info = {}
    for train, validation, end in cfg["folds"]:
        base = original/str(end)
        inputs = Inputs(base, base/"C1", base/"C1.npz", (train,validation,end))
        _, real = inputs.observed(train, teacher=True)
        assert (~real["teacher_valid"] | real["active"]).all()
        assert inputs.paths[train][-1][-1]["date"] < str(train+1)
        assert len({p[0]["episode"] for ps in inputs.paths.values() for p in ps}) == sum(map(len,inputs.paths.values()))
        sim, _ = inputs.simulation("single")
        initial = np.random.default_rng(cfg["origin_seed"]).integers(len(inputs.paths[train]),size=len(sim["f"]))[:10]
        a = batch(sim, slice(0,10), True)
        b = batch(real, initial, True)
        assert torch.equal(a["history"][:,0],b["history"][:,0])
        assert torch.equal(a["history_valid"][:,0],b["history_valid"][:,0])
        for wealth in (False,True):
            torch.manual_seed(0)
            model = make_policy("wide",wealth,True)
            with torch.no_grad():
                old = rollout(model,a,.0005)[0]
                fast = simulated_pnl(model,a,.0005)
            torch.testing.assert_close(fast,old,atol=2e-6,rtol=0)
        info[str(end)] = dict(paths={str(k):len(v) for k,v in inputs.paths.items()},
                             teacher_valid=int(real["teacher_valid"].sum()),
                             teacher_decisions=int(real["active"].sum()))
        for where in (base,root/"R1"/str(end)):
            repfile = where/"C1.npz"
            if not repfile.exists():
                continue
            with np.load(repfile) as data:
                k = np.linspace(.5,1.5,201)
                for term in (0,1,3,7,14,45,90):
                    t = np.full_like(k,term/365)
                    xi = np.broadcast_to(data["xi"][100],(len(k),2))
                    scale = np.ones(2)
                    c, dk, df = curve(t,k,xi,data,scale)
                    assert np.min(c-np.maximum(1-k,0)) > -1e-8
                    assert np.max(c) <= 1+1e-8
                    assert np.max(np.diff(c)) <= 1e-8
                    assert np.min(np.diff(c,n=2)) >= -1e-8
                    if not term:
                        np.testing.assert_array_equal(c,np.maximum(1-k,0))
                        continue
                    # Test analytic gradients away from the payoff/floor and grid kinks.
                    away = np.min(abs(k[:,None]-np.r_[np.unique(data["grid_k"]),1.]),axis=1) > 1e-5
                    eps = 1e-7
                    cp = curve(t,k+eps,xi,data,scale)[0]
                    cm = curve(t,k-eps,xi,data,scale)[0]
                    np.testing.assert_allclose((cp[away]-cm[away])/(2*eps),dk[away],atol=1e-6)
                    for j in range(2):
                        z = np.eye(2)[j]*eps
                        cp = curve(t,k,xi+z,data,scale)[0]
                        cm = curve(t,k,xi-z,data,scale)[0]
                        np.testing.assert_allclose((cp[away]-cm[away])/(2*eps),df[away,j],atol=1e-6)
    assert list(info["2025"]["paths"].values()) == [2626,387,773]
    delta, valid = bs_delta(np.ones(3), np.ones(3), np.array([.1,.0,1.1]), np.full(3,.1))
    assert valid.tolist() == [True,False,False] and .54 < delta[0] < .56
    result = dict(legacy_original_18_max_error=worst, legacy_account_precision_gap=rounding,
                  registered=1776, folds=info)
    write_json(root/"verification.json", result)
    print("Expanded study verification:",json.dumps(result),flush=True)

