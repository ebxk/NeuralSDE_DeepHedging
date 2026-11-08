"""Train frozen jobs with path-level risk and optional train-only guidance."""
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from contract_ledger import ledger
from prepare_common import digest, write_json
from train import adversary, batches, expected_shortfall, make_model, make_real, replay, score
from extended_data import batch
from extended_policy import load_legacy, make_policy, rollout, simulated_pnl


GATES = {"state_gate", "reliable_gate"}
DELTA = {"delta_input", "delta_residual"}


def needs_history(job):
    return job.h or job.policy in {"flat", "gru"}


def make_model_for(job, experts=None, fixed=None):
    return make_policy(job.policy, wealth=job.w, history=job.h, experts=experts, fixed=fixed)


def state_file(root, job):
    return root/"policies"/f"{job.key}.pt"


def expert_jobs(job):
    return {kind: replace(job, policy=kind) for kind in ("base", "wide")}


def restore(root, job, fixed=None):
    experts = None
    if job.policy in GATES | {"static"}:
        experts = {k: restore(root, j) for k, j in expert_jobs(job).items()}
    model = make_model_for(job, experts, fixed)
    if fixed is None:
        model.load_state_dict(torch.load(state_file(root, job), weights_only=True))
    return model


def risk_terms(z, lam, eta):
    loss = z.square()
    if lam:
        loss = loss + lam*(eta + torch.relu(-z-eta)/.05)
    return loss


def teacher_loss(ratio, data):
    mask = data["teacher_valid"]
    per_path = ((ratio-data["teacher"]).square()*mask).sum(1) / mask.sum(1).clamp(min=1)
    return per_path.sum() / mask.any(1).sum().clamp(min=1)


def norm(loss, parameters):
    return torch.stack([p.detach().square().sum() for p in
                        torch.autograd.grad(loss, parameters, allow_unused=True)
                        if p is not None]).sum().sqrt().item()


@torch.no_grad()
def predictions(model, job, data, size=512):
    pnl, ratios, gates = [], [], []
    for i in range(0, len(data["f"]), size):
        x = batch(data, slice(i, i+size), needs_history(job))
        p, r, g = rollout(model, x, job.cost/10000, capture=True)
        pnl.append(p.detach().numpy())
        ratios.append(r.detach().numpy())
        if g is not None:
            gates.append(g.detach().numpy())
    return np.concatenate(pnl), np.concatenate(ratios), np.concatenate(gates) if gates else None


def fit(model, job, cfg, inputs):
    start = time.perf_counter()
    simulated, ids = inputs.simulation(job.market, job.policy in DELTA)
    n = cfg["simulation_train"]
    assert len(ids) == n + cfg["simulation_valid"]
    _, val = inputs.observed(inputs.folds[1], job.policy in DELTA)
    guidance = job.t != "off"
    if guidance:
        _, real = inputs.observed(inputs.folds[0], job.policy in DELTA, teacher=job.t != "real")
    eta = torch.nn.Parameter(torch.tensor(1.))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params+([eta] if job.lam else []),
                           lr=.001 if job.policy in GATES else .003)
    sim_rng = np.random.default_rng(job.seed)
    real_rng = np.random.default_rng(job.seed+8401)
    levels = np.unique(ids[:n])
    fee = job.cost/10000

    def sim_loss(idx):
        p = simulated_pnl(model, batch(simulated, idx, needs_history(job)), fee)
        terms = risk_terms(100*p, job.lam, eta)
        risks = torch.stack([terms[ids[idx] == j].mean() for j in levels])
        q = adversary(risks.detach().numpy(), .1 if job.market == "robust" else 0)
        return (risks*torch.as_tensor(q, dtype=risks.dtype)).sum(), risks, q

    def add_loss(idx):
        data = batch(real, idx, needs_history(job))
        p, r, _ = rollout(model, data, fee)
        return (risk_terms(100*p, job.lam, eta).mean() if job.t == "real"
                else teacher_loss(r, data))

    beta = 0.
    if guidance and job.t != "warm":
        sim_cal = np.random.default_rng(job.seed+548)
        real_cal = np.random.default_rng(job.seed+549)
        a, b = [], []
        for idx in list(batches(ids[:n], cfg["batch_size"], sim_cal))[:cfg["gradient_batches"]]:
            a.append(norm(sim_loss(idx)[0], params))
            b.append(norm(add_loss(real_cal.integers(len(real["f"]), size=cfg["real_batch_size"])), params))
        beta = float(np.clip(.1*np.mean(a)/max(np.mean(b), 1e-8), 1e-4, 1e4))

    best, stale, trace = float("inf"), 0, []
    updates = 0
    for epoch in range(1, cfg["epochs"]+1):
        train_risk, train_q = [], []
        pretrain = job.t == "warm" and epoch <= cfg["warm_epochs"]
        alpha = beta
        if job.t in {"anneal", "real"}:
            alpha *= max(0., 1-(epoch-1)/max(cfg["teacher_epochs"]-1, 1))
        for idx in batches(ids[:n], cfg["batch_size"], sim_rng):
            opt.zero_grad(set_to_none=True)
            if pretrain or alpha:
                ridx = real_rng.integers(len(real["f"]), size=cfg["real_batch_size"])
                extra = add_loss(ridx)
            if pretrain:
                loss = extra
            else:
                loss, risks, q = sim_loss(idx)
                train_risk.append(risks.detach().numpy())
                train_q.append(q)
                if alpha:
                    loss = loss + alpha*extra
                    if job.t == "real":
                        loss = loss/(1+alpha)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite objective at epoch {epoch}")
            loss.backward()
            opt.step()
            updates += 1
        if epoch % cfg["validation_every"]:
            continue
        p = predictions(model, job, val)[0]
        value = score(p, job.lam)
        if not np.isfinite(value):
            raise FloatingPointError(f"nonfinite validation at epoch {epoch}")
        with torch.no_grad():
            # Keep memory bounded and retain equal prior weight per market.
            pred = torch.cat([simulated_pnl(model, batch(simulated, slice(i,i+512), needs_history(job)), fee)
                              for i in range(n,len(ids),512)])
            terms = risk_terms(100*pred, job.lam, eta).numpy()
            risk = np.array([terms[ids[n:] == j].mean() for j in levels])
            q = adversary(risk, .1 if job.market == "robust" else 0)
        trace.append(dict(epoch=epoch, updates=updates, validation=value,
                          rmse=float(np.sqrt(np.mean(p*p))), cvar=float(expected_shortfall(-p)),
                          beta=alpha, eta=float(eta.detach()),
                          train_risk=np.mean(train_risk, axis=0).tolist() if train_risk else [],
                          train_q=np.mean(train_q, axis=0).tolist() if train_q else [],
                          sim_risk=risk.tolist(), sim_q=q.tolist(),
                          sim_kl=float(np.sum(q*np.log(np.maximum(q*len(q),1e-300)))),
                          effective_markets=float(1/np.sum(q*q))))
        if value < best:
            best, stale, best_epoch = value, 0, epoch
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= cfg["patience"]:
            break
    return state, dict(epoch=best_epoch, best=best, seconds=time.perf_counter()-start,
                       trace=trace, updates=updates, beta=beta,
                       teacher_valid=int(real["teacher_valid"].sum()) if guidance and job.t != "real" else None,
                       teacher_decisions=int(real["active"].sum()) if guidance and job.t != "real" else None)


def legacy(job, model, inputs, original, parent, episodes_sha, cfg):
    if job.r or job.w or job.h or job.t != "off" or job.policy not in {"wide", "narrow"}:
        return
    name = {"wide": "same_capacity", "narrow": "same_inputs"}[job.policy]
    group = original/str(job.end)/"policies"/f"{job.rep}_{job.market}_{job.cost}bp_l{job.lam:g}_s{job.seed}"
    weights = group/f"{name}.pt"
    status = group/f"{name}.json"
    if not weights.exists() or not status.exists():
        return
    r = json.loads(status.read_text(encoding="utf8"))
    source = r.get("identity", {})
    old_cfg = json.loads((original/str(job.end)/"config.json").read_text(encoding="utf8"))
    keys = ("simulation_train", "simulation_valid", "batch_size", "epochs",
            "validation_every", "patience", "folds", "seeds", "lambdas", "origin_seed", "shock_seed")
    if (r.get("status") != "complete" or source.get("parent") != parent
        or source.get("episodes") != episodes_sha or digest(weights) != r.get("weights_sha256")
        or any(cfg[k] != old_cfg[k] for k in keys)):
        return
    state = torch.load(weights, weights_only=True)
    load_legacy(model, state)
    old = make_model(name)
    old.load_state_dict(state)
    paths, data = inputs.observed(inputs.folds[1])
    old_pnl = replay(old, paths, make_real(paths, inputs.common, inputs.episode_file), job.cost/10000)[0]
    p, ratios, _ = predictions(model, job, data)
    compatible = np.array([float(ledger(path, ratios[i,:len(path)-1], float(path[0]["option_close"]),
                        job.cost/10000)[-1]["wealth"])/(float(path[0]["spot"])*int(path[0]["unit"]))
                        for i,path in enumerate(paths)])
    error = float(np.max(np.abs(compatible-old_pnl)))
    if error > 1e-8:
        raise AssertionError(f"Legacy replay mismatch {job.key}: {error}")
    return model.state_dict(), {**r, "legacy_error": error, "legacy_weights_sha256": digest(weights),
                                "legacy_account_precision_gap": float(np.max(abs(p-compatible))),
                                "legacy_best": r["best"], "best": score(p,job.lam), "reused": True}


def complete(root, job, identity):
    weights = state_file(root, job)
    status = weights.with_suffix(".json")
    if not weights.exists() or not status.exists():
        return False
    r = json.loads(status.read_text(encoding="utf8"))
    return (r.get("status") == "complete" and r.get("identity") == identity
            and r.get("weights_sha256") == digest(weights))


def train_job(root, job, cfg, inputs, original, source):
    torch.manual_seed(job.seed)
    weights = state_file(root, job)
    identity = {**source, "job": job.meta()}
    experts = None
    if job.policy in GATES | {"capacity"}:
        parents = expert_jobs(job)
        identity["experts"] = {}
        for k, parent in parents.items():
            checkpoint = state_file(root, parent)
            status = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf8"))
            sha = digest(checkpoint)
            if (status.get("status") != "complete" or status.get("weights_sha256") != sha
                    or any(status.get("identity",{}).get(key) != value for key,value in source.items())):
                raise ValueError(f"Expert must be complete in the current run: {parent.key}")
            identity["experts"][k] = sha
        experts = {k: restore(root, j) for k, j in parents.items()}
    if complete(root, job, identity):
        return "cached"
    model = make_model_for(job, experts)
    fitted = legacy(job, model, inputs, original, source["market"], source["episodes"], cfg)
    state, stats = fitted if fitted else fit(model, job, cfg, inputs)
    temp = weights.with_suffix(".tmp")
    torch.save(state, temp)
    temp.replace(weights)
    model.load_state_dict(torch.load(weights, weights_only=True))
    _, data = inputs.observed(inputs.folds[1], job.policy in DELTA)
    value = score(predictions(model, job, data)[0], job.lam)
    if abs(value-stats["best"]) > (1e-6 if fitted else 1e-10):
        raise AssertionError(f"Saved model changed validation risk {value} vs {stats['best']}")
    write_json(weights.with_suffix(".json"), dict(**{k: v for k, v in stats.items() if k not in
               {"identity", "status", "parameters", "weights_sha256"}}, status="complete",
               identity=identity, parameters=sum(p.numel() for p in model.parameters()),
               weights_sha256=digest(weights)))
    return "reused" if fitted else "trained"

