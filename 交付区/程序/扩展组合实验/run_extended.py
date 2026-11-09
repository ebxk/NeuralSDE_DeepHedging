"""Run, resume, and verify the fixed expanded study."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import torch

from prepare_common import digest, write_json
from extended_data import Inputs
from extended_registry import registry
from extended_replay import baselines, folders, replay, statics
from extended_train import GATES, expert_jobs, state_file, train_job


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sources(project, root, context):
    end, r, rep, market = context
    old = project/"工作区/数据/处理后/创新组合实验"
    fold = old/str(end)
    if r == 0:
        where = fold
    else:
        where = root/"R1"/str(end)
    directory, representation = where/rep, where/f"{rep}.npz"
    marker = directory/"market.done.json"
    manifest = load(marker)
    for name, sha in manifest["outputs"].items():
        if digest(directory/name) != sha:
            raise ValueError(f"Market artifact changed: {directory/name}")
    episode_marker = fold/"episodes.done.json"
    if digest(fold/"episode_days.csv") != load(episode_marker)["outputs"]["episode_days.csv"]:
        raise ValueError("Episode rows changed")
    pool = directory/"pool.npz"
    if market != "single" and not pool.exists():
        if load(directory/"market.json")["members"] < 2:
            return None, None
        raise FileNotFoundError(pool)
    return (fold, directory, representation), dict(market=digest(marker),
                episodes=digest(episode_marker), representation=digest(representation))


def execute(project, root, cfg, token, context, jobs, frozen=None):
    torch.set_num_threads(1)
    phase = "train" if frozen is None else "replay"
    original = project/"工作区/数据/处理后/创新组合实验"
    try:
        files, source = sources(project, root, context)
        if files is None:
            return [dict(**j.meta(), key=j.key, status="unavailable",
                         reason="Fewer than two calibrated market members") for j in jobs]
        inputs = Inputs(*files, next(f for f in cfg["folds"] if f[-1] == context[0]))
        source["source"] = token
    except Exception:
        return [dict(**j.meta(), key=j.key, status="failed", phase=phase,
                     reason=traceback.format_exc()) for j in jobs]
    results = []
    for j in jobs:
        try:
            if frozen is None:
                status = train_job(root, j, cfg, inputs, original, source)
                print(status, j.key, flush=True)
            elif j.key in frozen:
                weights = state_file(root, j)
                record = load(weights.with_suffix(".json"))
                if digest(weights) != frozen[j.key] or record["weights_sha256"] != frozen[j.key]:
                    raise ValueError("Checkpoint changed after freeze")
                replay(root, root/"evaluation", inputs, j, record)
            else:
                continue
        except Exception:
            reason = traceback.format_exc()
            print(phase, "failed", j.key, reason, flush=True)
            results.append(dict(**j.meta(), key=j.key, status="failed", phase=phase, reason=reason))
            if frozen is None:
                write_json(state_file(root, j).with_suffix(".json"), dict(status="failed", reason=reason))
    if frozen is not None:
        for job in jobs:
            if job.policy == "reliable_gate":
                try:
                    if all(x.key in frozen for x in expert_jobs(job).values()):
                        statics(root, root/"evaluation", inputs, job, source)
                except Exception:
                    results.append(dict(**job.meta(), key=job.key, status="baseline_unavailable",
                                        reason=traceback.format_exc(), phase="static"))
        try:
            baselines(root/"evaluation", inputs, context, project/"交付区/结果/创新组合实验")
        except Exception:
            results.append(dict(context=context, status="baseline_unavailable",
                                reason=traceback.format_exc(), phase="baseline"))
    return results


def run_phase(project, root, cfg, token, groups, workers, frozen=None):
    failures = []
    label = "training" if frozen is None else "replay"
    if frozen is None:
        # Bound the size of the largest work unit without reloading for each job.
        chunks = []
        for context, jobs in groups.items():
            for seed in cfg["seeds"]:
                chunks.append((context,[j for j in jobs if j.seed == seed]))
        groups = chunks
    else:
        groups = list(groups.items())
    groups.sort(key=lambda item: -len(item[1]))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(execute, project, root, cfg, token, context, jobs, frozen):
                   (context, jobs) for context,jobs in groups}
        for future in as_completed(pending):
            context,jobs = pending[future]
            try:
                errors = future.result()
            except Exception:
                errors = [dict(**j.meta(), key=j.key, status="failed",
                               phase=label, reason=traceback.format_exc()) for j in jobs]
            failures.extend(errors)
            print(label, context, "complete", len(jobs), Counter(e["status"] for e in errors), flush=True)
    return failures


def freeze(root, jobs, token, failures):
    frozen = {}
    missing = {e["key"] for e in failures if "key" in e}
    for j in jobs:
        if j.key in missing:
            continue
        weights = state_file(root, j)
        status = weights.with_suffix(".json")
        if status.exists() and weights.exists():
            s = load(status)
            if (s.get("status") == "complete" and s["identity"]["source"] == token
                    and s["weights_sha256"] == digest(weights)):
                frozen[j.key] = s["weights_sha256"]
                continue
        failures.append(dict(**j.meta(), key=j.key, status="failed",
                             phase="freeze", reason="Checkpoint missing or stale"))
    return frozen


def archive_replays(root, frozen, include_aux=True):
    """Retain old auxiliary results separately before refreshing this run's set."""
    out = root/"evaluation"
    archive = root/"previous_replays"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    for file in (out/"paths").glob("*.json"):
        payload = load(file)
        registered = payload["meta"].get("registered", False)
        if (registered and file.stem in frozen) or (not registered and not include_aux):
            continue
        for path in (file, out/"daily"/f"{file.stem}.csv.gz"):
            if path.exists():
                target = archive/path.relative_to(out)
                target.parent.mkdir(parents=True, exist_ok=True)
                path.replace(target)


def run(args):
    from extended_report import report
    from extended_delivery import deliver
    from verify_extended import verify
    here = Path(__file__).resolve().parent
    project, root = args.project.resolve(), args.out.resolve()
    (root/"policies").mkdir(parents=True, exist_ok=True)
    out = root/"evaluation"
    folders(out)
    cfg = load(here/"study.json")
    jobs = registry(cfg)
    token = dict(torch=torch.__version__, numpy=np.__version__, files={p.name:digest(p) for p in
        [here/"study.json", *[here/f"{n}.py" for n in
         ("extended_registry","extended_train","extended_data","extended_curve","extended_policy")]]},
        parents={f"{folder}/{n}":digest(here.parent/folder/f"{n}.py")
                 for folder, modules in (("创新组合实验",("train","policy","prepare_common")),
                                         ("完整认购",("contract_ledger","hedge_experiment")))
                 for n in modules})
    write_json(root/"registry.json", dict(source=token, jobs={k:j.meta() for k,j in jobs.items()}))
    verify(project, root, cfg)
    if args.verify_only:
        return
    groups = defaultdict(list)
    for j in jobs.values():
        groups[j.end,j.r,j.rep,j.market].append(j)
    for js in groups.values():
        js.sort(key=lambda j: (j.seed, j.lam, j.policy in GATES|{"capacity"}, j.family != "main",
                               j.w or j.h, j.t != "off"))
    failures = run_phase(project, root, cfg, token, groups, args.workers)
    frozen = freeze(root, jobs.values(), token, failures)
    write_json(out/"frozen_checkpoints.json", frozen)
    write_json(out/"training_failures.json", failures)
    archive_replays(root, frozen)
    replay_failures = run_phase(project, root, cfg, token, groups, args.workers, frozen)
    for e in replay_failures:
        if e["status"] == "failed":
            frozen.pop(e["key"], None)
    write_json(out/"frozen_checkpoints.json", frozen)
    archive_replays(root, frozen, include_aux=False)
    failures = list({(e.get("key"),e["status"],e.get("phase"),str(e.get("context"))):e
                     for e in failures + replay_failures}.values())
    write_json(out/"failures.json", failures)
    completion = report(root, project, cfg, failures, expected=len(jobs))
    if not isinstance(completion, dict):
        completion = load(out/"completion.json")
    if completion["evaluated"] + completion.get("unavailable", 0) == len(jobs) and not completion.get("failures"):
        deliver(root, project, publish=args.publish)
    else:
        raise RuntimeError(f"See failures.json: {completion}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--workers", default=6, type=int)
    p.add_argument("--publish", action="store_true")
    p.add_argument("--verify-only", action="store_true")
    run(p.parse_args())

