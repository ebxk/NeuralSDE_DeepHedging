"""Frozen policy replay and corrected model sensitivities."""
from dataclasses import replace
import json
import os
from pathlib import Path
import time

import numpy as np

from prepare_common import digest, write_json
from report import baseline_book, bs_path, export
from train import score
from extended_curve import curve
from extended_data import books
from extended_train import DELTA, GATES, expert_jobs, predictions, restore, state_file


def folders(out):
    for name in ("paths", "daily"):
        (out/name).mkdir(parents=True, exist_ok=True)


def replay(root, out, inputs, job, record, fixed=None, tag=None):
    model = restore(root, job, fixed)
    paths, data = inputs.observed(job.end, job.policy in DELTA)
    key = job.key if tag is None else f"{job.key}_{tag}"
    dest = out/"paths"/f"{key}.json"
    daily = out/"daily"/f"{key}.csv.gz"
    mark = dict(weights=record.get("weights_sha256"), identity=record.get("identity"), fixed=fixed,
                replay_source=digest(__file__), export_source=digest(Path(export.__code__.co_filename)))
    if dest.exists() and daily.exists():
        cached = json.loads(dest.read_text(encoding="utf8"))
        if cached["meta"].get("frozen") == mark and cached["daily_sha256"] == digest(daily):
            return dest
    start = time.perf_counter()
    pnl, ratios, gate = predictions(model, job, data)
    ledger = books(paths, ratios, gate, job.cost)
    check = np.array([b[-1]["wealth"]/(float(p[0]["spot"])*int(p[0]["unit"])) for p,b in zip(paths,ledger)])
    error = float(np.max(np.abs(pnl-check)))
    if error > 1e-8:
        raise AssertionError(f"Differentiable ledger mismatch for {job.key}: {error:g}")
    seconds = record.get("seconds", 0.)
    dependencies = 0.
    if job.policy in GATES | {"static"}:
        dependencies = sum(json.loads(state_file(root, j).with_suffix(".json").read_text(
            encoding="utf8")).get("seconds", 0.) for j in expert_jobs(job).values())
    meta = dict(**job.meta(), registered=tag is None, frozen=mark, ledger_error=error,
                parameters=sum(p.numel() for p in model.parameters()), seconds=seconds,
                standalone_seconds=seconds+dependencies, replay_seconds=time.perf_counter()-start,
                date_to_xi=dict(zip(inputs.common["real_dates"], inputs.common["real_xi"])),
                reused=record.get("reused", False), updates=record.get("updates"))
    if tag is not None:
        meta["policy"] = tag
    return export(out, key, paths, ledger, job.cost, meta)


def statics(root, out, inputs, job, source):
    # Both fixed and selected weights use the same actual inventory and wealth.
    if not all(state_file(root, j).is_file() for j in expert_jobs(job).values()):
        return
    job = replace(job, policy="static", family="bridge")
    identity = dict(**source, experts={k: digest(state_file(root,j)) for k,j in expert_jobs(job).items()})
    _, val = inputs.observed(inputs.folds[1])
    scores = [score(predictions(restore(root,job,x/10),job,val)[0],job.lam) for x in range(11)]
    chosen = float(np.argmin(scores)/10)
    write_json((root/"policies"/f"{job.key}.json"), dict(identity=identity, score=scores, weight=chosen))
    for name, weight in (("fixed50", .5), ("static", chosen)):
        replay(root, out, inputs, job, dict(identity=identity), fixed=weight, tag=name)


def sensitivities(paths, inputs):
    dates = {d:i for i,d in enumerate(inputs.common["real_dates"])}
    rows = [row for path in paths for row in path[:-1]]
    i = np.array([dates[r["date"]] for r in rows])
    s = np.array([float(r["spot"]) for r in rows])
    k = np.array([float(r["strike"]) for r in rows])
    tau = (np.array([r["expiry"] for r in rows], dtype="datetime64[D]")-
           np.array([r["date"] for r in rows], dtype="datetime64[D]")).astype(float)/365
    c, dc, df = curve(tau, k/s, inputs.common["real_xi"][i], inputs.rep, inputs.common["scale"])
    d = c-(k/s)*dc
    cov = inputs.common["real_cov"][i]
    assert np.isfinite(cov).all() and (cov[:,0,0] > 1e-12).all()
    adjustment = np.sum(df*cov[:,1:,0], axis=1)/cov[:,0,0]
    assert np.isfinite(d+adjustment).all()
    cut = np.cumsum([len(path)-1 for path in paths])[:-1]
    return tuple(np.split(x, cut) for x in (d, adjustment))


def baselines(out, inputs, context, published):
    end, r, rep, market = context
    if market != "single":
        return
    paths = inputs.paths[end]
    ds, adjust = sensitivities(paths, inputs)
    for cost in (0,5,10) if end == 2025 else (5,):
        methods = {"Curve Delta": [np.clip(x,0,1) for x in ds],
                   "Local MV corrected": [np.clip(d+a,0,1) for d,a in zip(ds,adjust)]}
        if r == 0 and rep == "C0":
            methods["BS Delta"] = [bs_path(p) for p in paths]
        for name, ratios in methods.items():
            ledger = baseline_book(paths, ratios, cost)
            if name != "BS Delta":
                for book, delta, adj in zip(ledger, ds, adjust):
                    for i,row in enumerate(book):
                        row["model_delta"] = float(delta[i]) if i < len(delta) else ""
                        row["covariance_correction"] = float(adj[i]) if i < len(adj) else ""
                        row["unclipped_ratio"] = (float(delta[i]+(adj[i] if name == "Local MV corrected" else 0))
                                                  if i < len(adj) else "")
            meta = dict(end=end,r=r,rep=rep,t="off",w=False,h=False,cost=cost,market="baseline",policy=name,
                        family="baseline",lam=-1,seed=-1,registered=False,parameters=0,seconds=0,
                        standalone_seconds=0,replay_seconds=0,
                        date_to_xi=dict(zip(inputs.common["real_dates"],inputs.common["real_xi"])))
            export(out, f"{end}_R{r}_{rep}_{cost}bp_{name.replace(' ','_')}", paths, ledger, cost, meta)
    if r:
        return
    # Retain exact legacy ledgers with the same paths and a separate label.
    catalog = json.loads((published/"paths_index.json").read_text(encoding="utf8"))
    for x in catalog:
        if x["end"] != end or x["rep"] != rep:
            continue
        role = None
        if x["policy"] == "Local MV":
            role = "Local MV legacy"
        elif x["rep"] == "C0" and x["market"] == "gbm" and x["policy"] == "base":
            role = "GBM DH legacy"
        elif x["rep"] == "C0" and x["market"] == "single" and x["policy"] in {"base","factor"}:
            role = {"base":"Neural SDE DH legacy","factor":"Neural SDE DH factors legacy"}[x["policy"]]
        if role is None:
            continue
        name = x["id"]
        old = json.loads((published/"paths"/f"{name}.json").read_text(encoding="utf8"))
        assert [row["episode"] for row in old["rows"]] == [p[0]["episode"] for p in paths]
        source = published/"daily"/f"{name}.csv.gz"
        dest = out/"daily"/f"legacy_{name}.csv.gz"
        if not dest.exists():
            os.link(source, dest)
        assert digest(dest) == old["daily_sha256"]
        old["meta"].update(r=0,t="off",w=False,h=False,registered=False,family="baseline",policy=role)
        write_json(out/"paths"/f"legacy_{name}.json", old)

