"""Summarize frozen replays; all inference is paired at the expiry-month level."""
import csv
from collections import defaultdict
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np


META = ("end", "cost", "lam", "rep", "market", "policy", "r", "t", "w", "h", "family")
MEASURES = ("rmse", "cvar95", "mean_pnl", "trade_cost", "turnover")
NUMBERS = (*MEASURES, "std", "parameters", "seconds", "standalone_seconds", "replay_seconds")
REPORT_VERSION = 1


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")
    temporary.replace(path)


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def normalized(meta):
    result = dict(meta)
    for k, default in (("r", 0), ("t", "off"), ("w", False), ("h", False), ("family", "baseline")):
        result.setdefault(k, default)
    for k in ("end", "r", "seed"):
        result[k] = int(result.get(k, -1))
    for k in ("cost", "lam"):
        result[k] = float(result.get(k, -1))
    for k in ("w", "h"):
        result[k] = str(result[k]).lower() in {"true", "1"}
    return result


def name(meta, legacy=False):
    prefix = "legacy/" if legacy else ""
    return prefix + "/".join(str(meta[k]) for k in META[3:])


def expected_shortfall(loss):
    loss = np.sort(np.asarray(loss), axis=-1)[..., ::-1]
    mass = .05 * loss.shape[-1]
    n = int(np.ceil(mass))
    return (loss[..., :n] * np.clip(mass - np.arange(n), 0, 1)).sum(-1) / mass


def measure(values):
    pnl, cost, turnover = np.moveaxis(values, -1, 0)
    return np.stack((np.sqrt(np.mean(pnl * pnl, axis=-1)), expected_shortfall(-pnl),
                     pnl.mean(-1), cost.mean(-1), turnover.mean(-1)), axis=-1)


def load_records(files, legacy=False):
    records = []
    for file in files:
        payload = json.loads(file.read_text(encoding="utf8"))
        meta = normalized(payload["meta"])
        rows = sorted(payload["rows"], key=lambda row: str(row["episode"]))
        if not rows:
            raise ValueError(f"Empty replay: {file.name}")
        ids = [str(row["episode"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate path IDs: {file.name}")
        values = np.array([[row[k] for k in ("pnl", "cost", "turnover")] for row in rows], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite replay: {file.name}")
        records.append(dict(meta=meta, rows=rows, values=values, ids=ids,
                            months=np.array([row["expiry"][:7] for row in rows]),
                            file=file, legacy=legacy, method=name(meta, legacy)))
    return records


def summarize(records):
    seeds, catalog = [], []
    for record in records:
        m, rows, values = record["meta"], record["rows"], record["values"]
        groups = {"all": np.ones(len(rows), dtype=bool)}
        for key in ("term", "moneyness", "cross_div", "adjusted", "expiry"):
            labels = np.array([str(row[key])[:7] if key == "expiry" else str(row[key]) for row in rows])
            for label in np.unique(labels):
                groups[f"{key}={label}"] = labels == label
        for group, mask in groups.items():
            row = {k: m[k] for k in META}
            row.update(method=record["method"], seed=m["seed"], group=group, n=int(mask.sum()),
                       registered=bool(m.get("registered", False)))
            row.update(zip(MEASURES, map(float, measure(values[mask]))))
            row["std"] = float(values[mask, 0].std())
            for key in NUMBERS[len(MEASURES) + 1:]:
                row[key] = float(m.get(key, m.get("seconds", 0) if key == "standalone_seconds" else 0))
            gates = [float(x["gate_mean"]) for x, keep in zip(rows, mask) if keep and x.get("gate_mean", "") != ""]
            row["gate_mean"] = float(np.mean(gates)) if gates else ""
            seeds.append(row)
        catalog.append(dict(id=record["file"].stem, method=record["method"],
                            **{k: m[k] for k in (*META, "seed")},
                            daily=f"daily/{record['file'].stem}.csv.gz", paths=f"paths/{record['file'].name}"))
    grouped = defaultdict(list)
    for row in seeds:
        grouped[tuple(row[k] for k in (*META, "method", "group"))].append(row)
    summary = []
    for keys, group in grouped.items():
        row = dict(zip((*META, "method", "group"), keys), seeds=len(group), n=group[0]["n"])
        if len({x["n"] for x in group}) != 1 or len({x["seed"] for x in group}) != len(group):
            raise ValueError(f"Inconsistent seeds or sample counts: {keys}")
        for key in NUMBERS:
            values = [x[key] for x in group]
            row.update({key: float(np.mean(values)), key + "_min": min(values), key + "_max": max(values)})
        summary.append(row)
    return seeds, summary, catalog


def strategy_arrays(records):
    grouped = defaultdict(dict)
    meta = {}
    for r in records:
        m = r["meta"]
        key = (m["end"], m["cost"], m["lam"], r["method"])
        if m["seed"] in grouped[key]:
            raise ValueError(f"Duplicate replay identity: {key}, seed={m['seed']}")
        grouped[key][m["seed"]] = r
        meta[key] = m
    arrays = {}
    for key, seeds in grouped.items():
        if set(seeds) not in ({0, 1, 2}, {-1}):
            continue  # Partial seeds are reported, never promoted to a complete contrast.
        order = sorted(seeds)
        first = seeds[order[0]]
        if any(r["ids"] != first["ids"] or not np.array_equal(r["months"], first["months"]) for r in seeds.values()):
            raise ValueError(f"Unpaired random seeds: {key}")
        arrays[key] = (np.stack([seeds[s]["values"] for s in order]), first["ids"], first["months"])
    return arrays, meta


def contrasts(arrays, metadata):
    """Build comparisons from registered cells, without selecting by outcomes."""
    result, seen = [], set()

    def add(kind, label, coefficients):
        coefficients = {k: v for k, v in coefficients.items() if v}
        if not coefficients or not all(k in arrays for k in coefficients):
            return
        keys = list(coefficients)
        first = arrays[keys[0]]
        if any(arrays[k][1] != first[1] or not np.array_equal(arrays[k][2], first[2]) for k in keys):
            raise ValueError(f"Different path IDs in comparison: {label}")
        token = (kind, tuple(sorted(coefficients.items())))
        if token in seen:
            return
        seen.add(token)
        end, cost, lam = next(k[:3] for k in keys if k[2] >= 0)
        result.append(dict(end=end, cost=cost, lam=lam, kind=kind, contrast=label,
                           n=len(first[1]), coefficients=coefficients))

    current = [k for k in arrays if not k[3].startswith("legacy/") and k[2] >= 0]
    by_root = defaultdict(list)
    for key in current:
        by_root[key[:3]].append(key)
    for root, keys in by_root.items():
        main = [k for k in keys if metadata[k]["family"] in {"main", "cost"} and metadata[k]["policy"] == "wide"
                and metadata[k]["market"] == "single" and metadata[k]["rep"] in {"C0", "C1", "C2"}
                and metadata[k]["t"] in {"off", "anneal"}]
        cells = {(metadata[k]["rep"], metadata[k]["r"], int(metadata[k]["t"] == "anneal"),
                  int(metadata[k]["w"]), int(metadata[k]["h"])): k for k in main}
        modules = ("representation", "R", "T", "W", "H")
        for a, b in combinations(sorted(cells), 2):
            add("factorial_pair", f"{b} minus {a}", {cells[b]: 1., cells[a]: -1.})
        for j, module in enumerate(modules[1:], 1):
            edges = [(cell, tuple(1 if i == j else v for i, v in enumerate(cell))) for cell in cells if not cell[j]]
            edges = [(a, b) for a, b in edges if b in cells]
            if edges:
                coef = defaultdict(float)
                for a, b in edges:
                    coef[cells[b]] += 1 / len(edges)
                    coef[cells[a]] -= 1 / len(edges)
                add("main_effect", f"{module}; average {len(edges)} matched edges", dict(coef))
            for rep in ("C0", "C1", "C2"):
                full = (rep, 1, 1, 1, 1)
                without = tuple(0 if i == j else v for i, v in enumerate(full))
                if full in cells and without in cells:
                    add("leave_one_out", f"{rep} full minus without {module}", {cells[full]: 1., cells[without]: -1.})
        for i, j in combinations(range(1, 5), 2):
            squares = []
            for cell in cells:
                if cell[i] or cell[j]:
                    continue
                corners = [tuple((1 if k in active else v) for k, v in enumerate(cell))
                           for active in ((), (i,), (j,), (i, j))]
                if all(c in cells for c in corners):
                    squares.append(corners)
            if squares:
                coef = defaultdict(float)
                for square in squares:
                    for c, sign in zip(square, (1, -1, -1, 1)):
                        coef[cells[c]] += sign / len(squares)
                add("interaction", f"{modules[i]} x {modules[j]}; average {len(squares)} squares", dict(coef))
        # The three-level factor representation needs explicit pairwise contrasts.
        for low, high in combinations(("C0", "C1", "C2"), 2):
            edges = [(cell, (high, *cell[1:])) for cell in cells if cell[0] == low and (high, *cell[1:]) in cells]
            if edges:
                coef = defaultdict(float)
                for a, b in edges:
                    coef[cells[b]] += 1 / len(edges)
                    coef[cells[a]] -= 1 / len(edges)
                add("main_effect", f"{high} minus {low}; average {len(edges)} matched edges", dict(coef))
            full, without = (high, 1, 1, 1, 1), (low, 1, 1, 1, 1)
            if full in cells and without in cells:
                add("leave_one_out", f"full {high} minus {low}", {cells[full]: 1., cells[without]: -1.})
            for j in range(1, 5):
                squares = []
                for a, b in edges:
                    if a[j]:
                        continue
                    aa = tuple(1 if i == j else v for i, v in enumerate(a))
                    bb = tuple(1 if i == j else v for i, v in enumerate(b))
                    if aa in cells and bb in cells:
                        squares.append((a, b, aa, bb))
                if squares:
                    coef = defaultdict(float)
                    for square in squares:
                        for c, sign in zip(square, (1, -1, -1, 1)):
                            coef[cells[c]] += sign / len(squares)
                    add("interaction", f"({high}-{low}) x {modules[j]}; average {len(squares)} squares", dict(coef))
        # Holding every other feature fixed isolates individual mechanisms.
        for a in keys:
            ma = metadata[a]
            for b in keys:
                if a == b:
                    continue
                mb = metadata[b]
                equal = lambda fields: all(ma[f] == mb[f] for f in fields)
                state = ("rep", "market", "r", "t", "w", "h")
                if equal(("rep", "market", "policy", "r", "w", "h")) and ma["t"] != mb["t"] and ma["t"] != "off" and mb["t"] in {"off", "anneal"}:
                    add("teacher", f"{a[3]} minus {b[3]}", {a: 1., b: -1.})
                if equal(state) and ma["policy"] in {"narrow", "flat", "gru", "delta_input", "delta_residual"} and mb["policy"] == "wide":
                    kind = "capacity" if ma["policy"] == "narrow" else "history" if ma["policy"] in {"flat", "gru"} else "delta"
                    add(kind, f"{a[3]} minus {b[3]}", {a: 1., b: -1.})
                if equal(("rep", "market", "r", "t", "w", "h")) and ma["policy"] == "gru" and mb["policy"] == "flat":
                    add("history", f"{a[3]} minus {b[3]}", {a: 1., b: -1.})
                if equal(("rep", "market", "r", "t", "w", "h")) and ma["policy"] == "delta_residual" and mb["policy"] == "delta_input":
                    add("delta", f"{a[3]} minus {b[3]}", {a: 1., b: -1.})
                if equal(("rep", "market", "r", "t", "w", "h")) and ma["policy"] in {"state_gate", "reliable_gate"} and mb["policy"] in {"wide", "capacity", "fixed50", "static", "base", "state_gate"}:
                    add("A", f"{a[3]} minus {b[3]}", {a: 1., b: -1.})
                if equal(("rep", "policy", "r", "t", "w", "h")) and ma["market"] == "robust" and mb["market"] in {"single", "equal"}:
                    add("B", f"{a[3]} minus {b[3]}", {a: 1., b: -1.})
                if equal(("market", "policy", "r", "t", "w", "h")) and ma["rep"] == "C2" and mb["rep"] in {"C0", "C1"}:
                    add("C", f"{a[3]} minus {b[3]}", {a: 1., b: -1.})
            # Strong controls retained from the prior study, including its best wide network.
            for b in arrays:
                mb = metadata[b]
                if b[0:2] != root[0:2] or b[2] not in (root[2], -1):
                    continue
                legacy = b[3].startswith("legacy/")
                reference = (mb["market"] == "single" and (mb["rep"], mb["policy"]) in {
                    ("C0", "factor"), ("C0", "same_capacity"), ("C1", "same_capacity"), ("C2", "factor")})
                baseline = ((not legacy and mb["family"] == "baseline")
                            or mb["policy"] in {"BS Delta", "Local MV"} or mb["market"] == "gbm")
                if (legacy and reference) or baseline:
                    add("legacy" if legacy else "baseline", f"{a[3]} minus {b[3]}", {a: 1., b: -1.})
    return result


def month_draws(months, length, repetitions):
    labels = np.unique(months)
    n = len(labels)
    rng = np.random.default_rng(1105)
    length = min(length, n)
    starts = rng.integers(n - length + 1, size=(repetitions, int(np.ceil(n / length))))
    indices = (starts[..., None] + np.arange(length)).reshape(repetitions, -1)[:, :n]
    counts = np.zeros((repetitions, n), dtype=np.int16)
    for j in range(n):
        counts[:, j] = np.sum(indices == j, axis=1)
    return counts, np.searchsorted(labels, months)


def boot_metrics(values, months, length, repetitions):
    """Weighted draws avoid sorting the same losses 2,000 times per seed."""
    counts, month_index = month_draws(months, length, repetitions)
    output = np.zeros((repetitions, len(MEASURES)))
    for sample in values:
        loss_order = np.argsort(-sample[:, 0])[::-1]
        losses = -sample[loss_order, 0]
        for start in range(0, repetitions, 128):
            weights = counts[start:start + 128, month_index].astype(float)
            n = weights.sum(1)
            squared = weights @ (sample[:, 0] ** 2) / n
            tail_weights = weights[:, loss_order]
            before = np.cumsum(tail_weights, axis=1) - tail_weights
            tail = np.clip(.05 * n[:, None] - before, 0, tail_weights)
            es = tail @ losses / (.05 * n)
            means = weights @ sample / n[:, None]
            output[start:start + 128] += np.column_stack((np.sqrt(squared), es, means)) / len(values)
    return output


def infer(out, arrays, definitions, repetitions, identity):
    needed = sorted({k for item in definitions for k in item["coefficients"]})
    cache_dir = out / "bootstrap_cache"
    cache_dir.mkdir(exist_ok=True)
    seed_points = {k: measure(arrays[k][0]) for k in needed}
    points = {k: v.mean(0) for k, v in seed_points.items()}
    results = []
    for length in (1, 3, 6):
        cache = {}
        for key in needed:
            token = hashlib.sha256(json.dumps([identity, key, length, repetitions], sort_keys=True).encode()).hexdigest()
            path = cache_dir / (token + ".npz")
            if path.exists():
                with np.load(path) as saved:
                    cache[key] = saved["draws"]
            else:
                values, _, months = arrays[key]
                cache[key] = boot_metrics(values, months, length, repetitions)
                temporary = path.with_suffix(".tmp")
                with temporary.open("wb") as stream:
                    np.savez_compressed(stream, draws=cache[key])
                temporary.replace(path)
        for item in definitions:
            coef = item["coefficients"]
            point = sum(points[k] * v for k, v in coef.items())
            seed_delta = sum(seed_points[k] * v for k, v in coef.items())
            draws = sum(cache[k] * v for k, v in coef.items())
            lo, hi = np.quantile(draws, [.025, .975], axis=0)
            for j, metric in enumerate(MEASURES):
                row = {k: v for k, v in item.items() if k != "coefficients"}
                row.update(block_months=length, metric=metric, delta=float(point[j]), low=float(lo[j]), high=float(hi[j]),
                           terms=json.dumps({k[3]: v for k, v in coef.items()}, ensure_ascii=False),
                           repetitions=repetitions, seed_unit="mean_of_seedwise_metrics",
                           delta_seed_min=float(seed_delta[:, j].min()), delta_seed_max=float(seed_delta[:, j].max()),
                           favorable_seeds=int(np.sum(seed_delta[:, j] > 0 if metric == "mean_pnl" else seed_delta[:, j] < 0)),
                           seed_count=len(seed_delta))
                results.append(row)
    return results


def plots(out, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in summary if r["group"] == "all" and r["end"] == 2025 and r["cost"] == 5 and r["lam"] == 0]
    fig, ax = plt.subplots(figsize=(9, 6))
    families = sorted({r["family"] for r in rows})
    for family in families:
        subset = [r for r in rows if r["family"] == family]
        ax.scatter([100 * r["rmse"] for r in subset], [100 * r["cvar95"] for r in subset], label=family, alpha=.7)
    ax.set(xlabel="RMSE (%)", ylabel="CVaR95 loss (%)", title="All registered candidates: historical 2024-2025, 5bp, MSE")
    if families:
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "risk_tradeoff.png", dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ("rmse", "cvar95")):
        for rep in ("C0", "C1"):
            for full in (False, True):
                selected = sorted([r for r in summary if r["group"] == "all" and r["cost"] == 5 and r["lam"] == 0
                                   and r["rep"] == rep and r["market"] == "single" and r["policy"] == "wide"
                                   and r["r"] == int(full) and r["w"] == full and r["h"] == full
                                   and r["t"] == ("anneal" if full else "off")], key=lambda r: r["end"])
                ax.plot([r["end"] for r in selected], [100 * r[metric] for r in selected], marker="o", label=rep + (" full" if full else " plain"))
        ax.set(xlabel="Evaluation end year", ylabel=metric + " (%)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "rolling_stability.png", dpi=160); plt.close(fig)


def dashboard(out, summary, catalog, completion):
    # Embedded summaries work via file://. Detailed saved books are local links.
    data = json.dumps(dict(summary=summary, paths=catalog, completion=completion), ensure_ascii=False).replace("</", "<\\/")
    page = '''<!doctype html><meta charset="utf-8"><title>50ETF 扩展实验</title>
<style>body{font:15px system-ui;margin:32px;background:#f7f8fa;color:#18232b}select,input{padding:7px;margin:5px}table{border-collapse:collapse;background:white;width:100%}td,th{padding:8px;text-align:right;border-bottom:1px solid #ddd}td:first-child,th:first-child{text-align:left}.scroll{max-height:65vh;overflow:auto}th{position:sticky;top:0;background:#e9edf1}small{color:#556}</style>
<h1>50ETF 扩展组合实验</h1><p>2024–2025 为已观察历史评价。指标越小越好，但平均损益越大越好。种子不是额外市场样本。</p><p id="status"></p>
<label>截止年 <select id="end"></select></label><label>成本 <select id="cost"></select></label><label>目标 <select id="lam"></select></label><label>分组 <select id="group"></select></label><input id="query" placeholder="组合、表示或策略名称"><label>排序 <select id="metric"><option>rmse</option><option>cvar95</option><option>trade_cost</option><option>standalone_seconds</option></select></label>
<div class="scroll"><table><thead><tr><th>方法</th><th>样本/种子</th><th>RMSE</th><th>CVaR95</th><th>平均损益</th><th>成本</th><th>耗时(秒)</th></tr></thead><tbody id="rows"></tbody></table></div>
<h2>已保存的具体路径和逐日持仓</h2><small>链接只读取本地保存结果，不触发重新训练。路径 JSON 可查看合约编号和单条损益；逐日压缩账本含持仓、现金和成本。</small><div id="books"></div>
<p><a href="paired_intervals.csv">配对区间</a> · <a href="factorial_contrasts.csv">主效应与交互</a> · <a href="legacy_comparison.csv">历史强对照</a> · <a href="failures.json">失败与缺项</a></p>
<script>const DATA=__DATA__;const $=x=>document.getElementById(x);const escape=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
for(const k of ['end','cost','lam','group']){let v=[...new Set(DATA.summary.map(r=>r[k]))].sort((a,b)=>typeof a==='number'?a-b:String(a).localeCompare(String(b)));$(k).innerHTML=v.map(x=>`<option>${escape(x)}</option>`).join('');} $('end').value=2025;$('cost').value=5;$('lam').value=0;$('group').value='all';$('status').textContent=JSON.stringify(DATA.completion);
function draw(){let selected=DATA.summary.filter(r=>['end','cost','lam','group'].every(k=>String(r[k])===$(k).value)&&r.method.toLowerCase().includes($('query').value.toLowerCase())).sort((a,b)=>a[$('metric').value]-b[$('metric').value]);const pct=x=>(100*x).toFixed(4)+'%';$('rows').innerHTML=selected.map(r=>`<tr><td>${escape(r.method)}</td><td>${r.n}/${r.seeds}</td><td>${pct(r.rmse)}</td><td>${pct(r.cvar95)}</td><td>${pct(r.mean_pnl)}</td><td>${pct(r.trade_cost)}</td><td>${r.standalone_seconds.toFixed(1)}</td></tr>`).join('');const names=new Set(selected.map(r=>r.method));$('books').innerHTML=DATA.paths.filter(r=>names.has(r.method)&&String(r.end)===$('end').value&&String(r.cost)===$('cost').value&&String(r.lam)===$('lam').value).map(r=>`<p>${escape(r.method)} / seed ${r.seed} <a href="${encodeURI(r.paths)}">逐路径结果</a> <a href="${encodeURI(r.daily)}">逐日账本</a></p>`).join('');}for(const k of ['end','cost','lam','group','query','metric'])$(k).oninput=draw;draw();</script>'''
    (out / "index.html").write_text(page.replace("__DATA__", data), encoding="utf8")


def report(root, project, cfg, failures, expected=1776):
    root, project = Path(root), Path(project)
    out = root / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    files = sorted((out / "paths").glob("*.json"))
    if not files:
        raise ValueError("No frozen evaluation paths available")
    records = load_records(files)
    frozen_path = out / "frozen_checkpoints.json"
    stale = []
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text(encoding="utf8"))
        if not isinstance(frozen, dict):
            raise ValueError("Expected checkpoint manifest mapping job keys to hashes")
        stale = [r for r in records if r["meta"].get("registered", False) and
                 (r["file"].stem not in frozen or (r["meta"].get("frozen") and
                  r["meta"]["frozen"].get("weights") != frozen.get(r["file"].stem)))]
        stale_paths = {r["file"] for r in stale}
        records = [r for r in records if r["file"] not in stale_paths]
    # Ensure all strategies in a fold use exactly the same evaluation episodes.
    fold_ids = {}
    for r in records:
        end = r["meta"]["end"]
        if end in fold_ids and fold_ids[end] != r["ids"]:
            raise ValueError(f"Unfair evaluation samples in fold {end}")
        fold_ids[end] = r["ids"]
    seeds, summary, catalog = summarize(records)
    write_csv(out / "metrics_by_seed.csv", seeds)
    write_csv(out / "summary.csv", summary)
    write_json(out / "paths_index.json", catalog)
    old = project / "交付区" / "结果" / "创新组合实验"
    legacy = []
    if (old / "summary.csv").exists():
        with (old / "summary.csv").open(encoding="utf8", newline="") as stream:
            previous = list(csv.DictReader(stream))
        write_csv(out / "legacy_comparison.csv", [dict(source="previous", **r) for r in previous]
                  + [dict(source="extension", **r) for r in summary])
        candidates = [f for f in (old / "paths").glob("*.json") if any(s in f.name for s in
                      ("same_capacity", "_factor.json", "_gbm_", "_BS_Delta.json", "_Local_MV.json"))]
        legacy = load_records(candidates, legacy=True)
    else:
        write_csv(out / "legacy_comparison.csv", [dict(source="extension", **r) for r in summary])
    arrays, metadata = strategy_arrays(records + legacy)
    definitions = contrasts(arrays, metadata)
    bootstrap = cfg.get("bootstrap", 2000)
    repetitions = int(bootstrap.get("repetitions", 2000) if isinstance(bootstrap, dict) else bootstrap)
    identity = hashlib.sha256((str(REPORT_VERSION) + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
                              + json.dumps(cfg, sort_keys=True, ensure_ascii=False)
                              + "".join(hashlib.sha256(r["file"].read_bytes()).hexdigest() for r in records + legacy)).encode()).hexdigest()
    pairs = infer(out, arrays, definitions, repetitions, identity)
    write_csv(out / "paired_intervals.csv", pairs)
    write_csv(out / "factorial_contrasts.csv", [r for r in pairs if r["kind"] in {"main_effect", "interaction", "leave_one_out"}])
    write_csv(out / "mechanism_contrasts.csv", [r for r in pairs if r["kind"] in {"teacher", "capacity", "history", "delta", "A", "B", "C"}])
    registered = {tuple(r["meta"][k] for k in (*META, "seed")) for r in records if r["meta"].get("registered", False)}
    failure_rows = list(failures)
    unavailable = sum(f.get("status") == "unavailable" for f in failure_rows)
    failed = sum(f.get("status") not in {"unavailable", "baseline_unavailable"} for f in failure_rows)
    completion = dict(expected=int(expected), evaluated=len(registered), unavailable=unavailable, failures=failed,
                      baseline_unavailable=sum(f.get("status") == "baseline_unavailable" for f in failure_rows),
                      missing=max(0, int(expected) - len(registered) - unavailable - failed),
                      replay_files=len(records), complete=len(registered) + unavailable == expected and failed == 0,
                      excluded_stale_replays=len(stale),
                      paired_definitions=len(definitions), report_identity=identity,
                      historical_evaluation="2024-2025 already observed", repetitions=repetitions)
    write_json(out / "completion.json", completion)
    write_json(out / "failures.json", failure_rows)
    write_json(out / "report_manifest.json", dict(identity=identity, version=REPORT_VERSION,
               files=len(files), bootstrap_seed=1105, blocks=[1, 3, 6], samples={str(k): len(v) for k, v in fold_ids.items()}))
    visual_summary = list(summary)
    visual_catalog = list(catalog)
    if legacy:
        _, legacy_summary, legacy_catalog = summarize(legacy)
        visual_summary.extend(r for r in legacy_summary if r["group"] == "all" and
                              r["market"] == "single" and r["policy"] == "same_capacity")
        for item in legacy_catalog:
            if item["market"] == "single" and item["policy"] == "same_capacity":
                item["daily"] = old.resolve().as_uri() + "/" + item["daily"]
                item["paths"] = old.resolve().as_uri() + "/" + item["paths"]
                visual_catalog.append(item)
        for r in visual_summary:
            if r["method"].startswith("legacy/"):
                r["family"] = "legacy wide controls"
    plots(out, visual_summary)
    dashboard(out, visual_summary, visual_catalog, completion)
    rows = sorted([r for r in summary if r["group"] == "all" and (r["end"], r["cost"], r["lam"]) == (2025, 5, 0)], key=lambda r: r["rmse"])
    lines = ["# 50ETF 扩展组合实验结果", "", f"登记训练评价完成 {len(registered)}/{expected}；不可运行 {unavailable}，失败 {failed}，待补 {completion['missing']}。", "",
             "2024–2025 是已观察的历史评价。下表包含强普通网络和所有可用组合，不把预设核心组合误称为全体最优。完整分组、种子范围、机制比较和历史强对照见 CSV 与本地展示页面。", "",
             "| 方法 | 样本/种子 | RMSE | CVaR95 损失 | 交易成本 |", "|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['method']} | {r['n']}/{r['seeds']} | {r['rmse']:.4%} | {r['cvar95']:.4%} | {r['trade_cost']:.4%} |")
    focus = [r for r in pairs if r["end"] == 2025 and r["cost"] == 5 and r["lam"] == 0
             and r["kind"] in {"main_effect","interaction"} and r["block_months"] == 3
             and r["metric"] in {"rmse","cvar95"}]
    if focus:
        lines += ["", "## 5bp 主时期的模块效应", "",
                  "| 预登记对照 | 指标 | 差值（百分点） | 95% 区间 | 同方向改善的滚动折 |",
                  "|---|---|---:|---:|---:|"]
        for row in focus:
            same = [p for p in pairs if p["end"] in (2021,2022,2023)
                    and all(p[k] == row[k] for k in ("cost","lam","kind","contrast","block_months","metric"))]
            interval = f"[{100*row['low']:+.4f}, {100*row['high']:+.4f}]"
            if row["low"] <= 0 <= row["high"]:
                interval += "（证据有限）"
            improving = sum(x["delta"] < 0 for x in same)
            lines.append(f"| {row['contrast']} | {row['metric']} | {100*row['delta']:+.4f} | {interval} | {improving}/{len(same)} |")
    lines += ["", "差值为候选减对照。RMSE、尾部损失和成本差值为负表示下降；平均损益差值为正表示增加。主效应平均所有匹配边，二阶交互为双模块效应减两个单模块效应，移除对照为完整组合减移除一个模块。", "",
              "区间按相同到期月份配对，分别使用 1、3、6 个月连续块，固定 2,000 次（调试配置可降低）重抽样。先在各种子内计算指标，再平均；种子不扩充市场样本。各区间未做多重比较校正，跨零时写证据有限，不宣布唯一总冠军。", "",
              "统计显著性不能修复已经观察过历史评价集的问题。连续无分红模拟、同收盘观测成交、离散边界修正等近似继续适用。策略耗时不等于共享市场模型加策略的全部研究成本。"]
    (out / "结果说明.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    return completion
