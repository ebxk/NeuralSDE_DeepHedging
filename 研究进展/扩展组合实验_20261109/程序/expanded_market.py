"""Prepare the R1 market artifacts without changing the frozen R0 files."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import traceback
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
UPSTREAM_REVISION = "6fa18790f14b56c99407ef987406459fa6e3b988"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def output_hashes(paths):
    from prepare_common import digest

    return {path.name: digest(path) for path in paths}


def reusable(marker, token):
    if not marker.is_file():
        return False
    try:
        record = read_json(marker)
        paths = [marker.parent / name for name in record["outputs"]]
        return record["identity"] == token and output_hashes(paths) == record["outputs"]
    except (KeyError, ValueError, OSError):
        return False


def mark(marker, token, paths, **status):
    from prepare_common import write_json

    write_json(marker, dict(identity=token, outputs=output_hashes(paths), **status))


def fit_market(factors, spot, upstream, days, out, config, pool):
    import market
    from prepare_common import write_json

    fit_member = market.fit_member

    class BootstrapNotRequested(ValueError):
        pass

    def single_member(*args):
        if args[-1]:
            raise BootstrapNotRequested()
        return fit_member(*args)

    if pool == "1":
        market.run(factors, spot, upstream, days, out, config)
    else:
        with patch.object(market, "fit_member", single_member):
            market.run(factors, spot, upstream, days, out, config)
    diagnostics_path = Path(out) / "market.json"
    diagnostics = read_json(diagnostics_path)
    if pool == "0":
        diagnostics["fit"] = [row for row in diagnostics["fit"] if row["seed"] == 2026]
    diagnostics["pool_requested"] = pool == "1"
    write_json(diagnostics_path, diagnostics)


def check_market(output, baseline, factor, diagnostics):
    import numpy as np

    with np.load(factor) as rep, np.load(output/"common.npz") as r1, np.load(baseline) as r0:
        for key in ("time", "warm", "strike", "premium", "real_dates", "return_dates", "returns"):
            np.testing.assert_array_equal(r1[key], r0[key], err_msg=key)
        np.testing.assert_array_equal(r1["real_dates"], rep["dates"])
        shape = r1["time"].shape
        for key in ("real_xi", "real_q", "gbm", "real_cov", "scale"):
            if not np.isfinite(r1[key]).all():
                raise ValueError(f"Nonfinite common array {key}")
    for name in ("single", "pool"):
        path = output/f"{name}.npz"
        if name == "pool" and diagnostics["members"] == 1:
            continue
        with np.load(path) as paths:
            if paths["spot"].shape != shape or paths["xi"].shape != (*shape, 2):
                raise ValueError(f"Unexpected simulated path shape for {name}")
            if not all(np.isfinite(paths[key]).all() for key in ("spot", "xi", "quality")):
                raise ValueError(f"Nonfinite simulated array in {name}")
    for row in diagnostics["numeric"]:
        if row["paired2"]["subdivisions"] != 2 or row["paired4"]["subdivisions"] != 4:
            raise ValueError("Missing paired two- and four-substep diagnostics")
        if min(row["full_slack"], row["fine_slack"]) < -1e-7:
            raise ValueError("Simulated factor path breaches static price constraints")


def prepare_market(factor, spot, upstream, days, output, config, baseline, token, pool, env):
    marker = output/"market.done.json"
    if reusable(marker, token):
        return read_json(marker)["status"]
    output.mkdir(exist_ok=True)
    marker.unlink(missing_ok=True)
    log = output/"market.log"
    with TemporaryDirectory(prefix="attempt-", dir=output) as temp, log.open("wb") as stream:
        staging = Path(temp)
        subprocess.run([sys.executable, "-u", str(HERE/"expanded_market.py"), "_fit",
                        str(factor), str(spot), str(upstream), str(days), str(staging),
                        str(config), str(int(pool))], stdout=stream, stderr=subprocess.STDOUT,
                       env=env, check=True)
        diagnostics = read_json(staging/"market.json")
        check_market(staging, baseline, factor, diagnostics)
        names = sorted(path.name for path in staging.iterdir())
        for name in names:
            (staging/name).replace(output/name)
        for path in [output/"pool.npz", *output.glob("market_*.weights.h5")]:
            if path.name not in names:
                path.unlink(missing_ok=True)
    pool_status = ("ready" if "pool.npz" in names else "unavailable") if pool else "not_requested"
    status = dict(single="ready", pool=pool_status, members=diagnostics["members"])
    mark(marker, token, [output/name for name in names], status=status)
    return status


def frozen_episodes(baseline, inputs, fold):
    days = baseline/"episode_days.csv"
    record = read_json(baseline/"episodes.done.json")
    baseline_inputs = record["identity"]["source"]["files"]
    if (record["identity"]["fold"] != fold
            or output_hashes([days])["episode_days.csv"] != record["outputs"]["episode_days.csv"]
            or any(baseline_inputs.get(name) != digest for name, digest in inputs.items())):
        raise ValueError(f"Frozen episode inputs do not match R0 for {fold}")
    return days


def prepare(args):
    project, out = args.project.resolve(), args.out.resolve()/"R1"
    work = project/"工作区"
    program = work/"当前研究"/"程序"
    original, legacy = program/"创新组合实验", program/"完整认购"
    upstream = work/"运行支持"/"临时文件"/"neuralSDE-marketmodel"
    baseline = work/"数据"/"处理后"/"创新组合实验"
    sys.path[:0] = [str(original), str(legacy)]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(map(str, (original, legacy))),
           "PYTHONUTF8": "1", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
           "MKL_NUM_THREADS": "2", "TF_NUM_INTEROP_THREADS": "2", "TF_CPP_MIN_LOG_LEVEL": "2"}
    from expanded_surfaces import SOURCE_FLAGS, fold_support, make_surfaces, read_quotes, source_report
    from prepare_common import write_json

    out.mkdir(parents=True, exist_ok=True)
    failure_file, status_file = out/"market_failures.json", out/"market_status.json"
    status = {}
    failures = []
    write_json(failure_file, failures)
    write_json(status_file, status)
    try:
        revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"],
                                           text=True).strip()
        if revision != UPSTREAM_REVISION:
            raise ValueError(f"Unexpected upstream revision {revision}")
        cfg = read_json(original/"study.json")
        raw_files = output_hashes([args.options, args.spot])
        surface = out/"surfaces.npz"
        marker = out/"surface.done.json"
        token = dict(revision=revision, folds=cfg["folds"], inputs=raw_files,
                     code=output_hashes([HERE/"expanded_surfaces.py", legacy/"contract_ledger.py"]))
        support_path = out/"support.json"
        if not reusable(marker, token):
            marker.unlink(missing_ok=True)
            quotes = read_quotes(args.options, args.spot)
            support = {str(fold[-1]): fold_support(quotes, fold) for fold in cfg["folds"]}
            make_surfaces(quotes, upstream, surface)
            for fold in cfg["folds"]:
                support[str(fold[-1])]["source_by_split"] = source_report(surface, fold)
            write_json(support_path, dict(source_flags=SOURCE_FLAGS, folds=support))
            mark(marker, token, [surface, support_path])
        support = read_json(support_path)["folds"]
    except Exception:
        write_json(failure_file, [dict(stage="surface", error=traceback.format_exc())])
        raise

    def failure(stage, end, exc, rep=None):
        detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        failures.append(dict(stage=stage, fold=end, representation=rep, error=detail))
        write_json(failure_file, failures)
        print(f"R1 {end}/{rep or stage}: {exc}", flush=True)

    dependencies = output_hashes([HERE/"expanded_market.py", original/"market.py",
                                  original/"representation.py", legacy/"bs_baseline.py",
                                  legacy/"contract_ledger.py"])
    surface_hash = read_json(out/"surface.done.json")["outputs"]["surfaces.npz"]
    tasks = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        for fold in cfg["folds"]:
            end = str(fold[-1])
            folder = out/end
            folder.mkdir(exist_ok=True)
            try:
                if not support[end]["passed"]:
                    raise ValueError("Training-only support threshold failed. See R1/support.json")
                days = frozen_episodes(baseline/end, raw_files, fold)
                fcfg = {**cfg, "split_ends": fold}
                config = folder/"config.json"
                write_json(config, fcfg)
                reps = [folder/f"C{i}.npz" for i in range(3)]
                marker = folder/"representation.done.json"
                token = dict(config=fcfg, revision=revision, dependencies=dependencies,
                             surface=surface_hash, episodes=output_hashes([days]))
                if not reusable(marker, token):
                    marker.unlink(missing_ok=True)
                    subprocess.run([sys.executable, "-u", str(original/"representation.py"),
                                    str(surface), str(days), str(upstream), str(folder), str(config)],
                                   env=env, check=True)
                    mark(marker, token, reps)
            except Exception as exc:
                failure("representation", end, exc)
                continue
            print(f"R1 {end}: representations ready", flush=True)
            for factor in (reps[1], reps[0], reps[2]):
                rep = factor.stem
                pool = end == "2025" and rep == "C1"
                r0 = baseline/end/rep/"common.npz"
                identity = dict(representation=token, factor=output_hashes([factor]),
                                spot=raw_files[args.spot.name], pool=pool,
                                baseline=output_hashes([r0]))
                future = executor.submit(prepare_market, factor, args.spot, upstream, days,
                                         folder/rep, config, r0, identity, pool, env)
                tasks[future] = end, rep
        for future in as_completed(tasks):
            end, rep = tasks[future]
            try:
                status[f"{end}/{rep}"] = future.result()
                write_json(status_file, status)
                print(f"R1 {end}/{rep}: {status[f'{end}/{rep}']}", flush=True)
            except Exception as exc:
                failure("market", end, exc, rep)
    return not failures


if __name__ == "__main__":
    if sys.argv[1:2] == ["_fit"]:
        fit_market(*sys.argv[2:])
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        for name in ("project", "out", "options", "spot"):
            parser.add_argument(f"--{name}", type=Path, required=True)
        sys.exit(0 if prepare(parser.parse_args()) else 1)
