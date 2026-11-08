"""Construct the fixed R1 price grid and audit its quote support."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from contract_ledger import terms


MATURITIES = np.array([7, 14, 30, 60, 90])
STRIKES = np.array([.85, .90, .95, 1., 1.05, 1.10, 1.15])
INTERPOLATED, ANCHOR, EXTRAPOLATED = 1, 2, 4
SOURCE_FLAGS = {
    "observed": 0, "interpolated": INTERPOLATED,
    "anchor": ANCHOR, "extrapolated": EXTRAPOLATED,
}


def read_quotes(option_file, spot_file):
    columns = ["date", "code", "trading_code", "list_date", "expire_date",
               "contract_type", "close", "volume", "position"]
    options = pd.read_csv(option_file, usecols=columns)
    spot = pd.read_csv(spot_file, encoding="gb18030").set_index("交易日期")["收盘价(元)"]
    options = options[(options.contract_type == "CO") & (options.close > 0)
                      & (options.volume > 0) & (options.position > 0)].copy()
    options["ttm"] = (pd.to_datetime(options.expire_date) -
                      pd.to_datetime(options.date)).dt.days
    options = options[options.ttm > 0].copy()
    options["spot"] = options.date.map(spot)
    if not np.isfinite(options.spot).all() or (options.spot <= 0).any():
        raise ValueError("Eligible option observations require a positive same-day spot close")
    options["strike"] = [float(terms(row, row["date"])[0]) for row in
                         options[["trading_code", "list_date", "expire_date", "date"]].to_dict("records")]
    options["k"] = options.strike / options.spot
    return options


def fold_support(quotes, fold):
    cutoff = f"{fold[0] + 1}-01-01"
    train = quotes[quotes.date < cutoff]
    days = train.date.nunique()
    if not days:
        raise ValueError(f"No eligible training quotes for {fold}")
    short = train.ttm.between(5, 9)
    short_days = int(train.loc[short, "date"].nunique())
    wing, nodes = [], []
    for k in STRIKES[[0, -1]]:
        near_k = (train.k - k).abs() <= .025 + 1e-12
        count = int(train.loc[near_k & train.ttm.between(1, 90), "date"].nunique())
        wing.append(dict(k=float(k), days=count, fraction=count/days, passed=count/days >= .50))
    for t in MATURITIES:
        for k in STRIKES:
            if t != 7 and k not in STRIKES[[0, -1]]:
                continue
            near = train[(train.ttm - t).abs().le(2) & (train.k - k).abs().le(.025 + 1e-12)]
            contracts = int(near.code.nunique())
            months = int(near.expire_date.str.slice(0, 7).nunique())
            nodes.append(dict(ttm=int(t), k=float(k), days=int(near.date.nunique()),
                              contracts=contracts, expiry_months=months,
                              passed=contracts >= 30 and months >= 6))
    tenor = dict(ttm=7, days=short_days, fraction=short_days/days,
                 passed=short_days/days >= .20)
    return dict(fold=fold, training_days=int(days), first_training_date=str(train.date.min()),
                last_training_date=str(train.date.max()), tenor=tenor, wings=wing, nodes=nodes,
                passed=tenor["passed"] and all(row["passed"] for row in wing + nodes),
                rules=dict(tenor_window_days=2, tenor_min_day_fraction=.20,
                           wing_moneyness_window=.025, wing_ttm_days=[1, 90],
                           wing_min_day_fraction=.50, node_min_contracts=30,
                           node_min_expiry_months=6, node_ttm_window_days=2,
                           node_moneyness_window=.025))


def quote_curve(quotes, spot, intrinsic):
    quotes = quotes.sort_values(["volume", "code"], ascending=[False, True])
    quotes = quotes.drop_duplicates("strike").sort_values("strike")
    k, c = quotes.k.to_numpy(), quotes.close.to_numpy()/spot
    slope = ((c[-1] - c[-2]) / (k[-1] - k[-2]) if len(k) > 1 else (c[-1] - 1) / k[-1])
    slope = np.clip(slope, -1, 0)
    curve = np.interp(STRIKES, np.r_[0., k], np.r_[1., c])
    right = STRIKES > k[-1]
    curve[right] = np.maximum(intrinsic[right], c[-1] + (STRIKES[right] - k[-1])*slope)
    exact = np.isclose(STRIKES[:, None], k, rtol=0, atol=1e-12).any(axis=1)
    source = np.full(len(STRIKES), INTERPOLATED, dtype=np.uint8)
    source[STRIKES < k[0]] = INTERPOLATED | ANCHOR | EXTRAPOLATED
    source[right] = EXTRAPOLATED
    source[exact] = 0
    return curve, source


def make_surfaces(quotes, upstream, output):
    sys.path.insert(0, str(upstream))
    from arbitragerepair.constraints import detect
    from arbitragerepair.repair import l1
    from cvxopt import solvers

    solvers.options["glpk"] = {"msg_lev": "GLP_MSG_OFF"}
    grid_t = np.repeat(MATURITIES/365, len(STRIKES))
    grid_k = np.tile(STRIKES, len(MATURITIES))
    intrinsic = np.maximum(1 - STRIKES, 0)
    days, spots, raw, repaired, sources = [], [], [], [], []
    for day, group in quotes.groupby("date", sort=True):
        spot = group.spot.iat[0]
        times, curves = [0], [intrinsic]
        provenance = [np.full(len(STRIKES), ANCHOR, dtype=np.uint8)]
        for ttm, quotes_at_t in group.groupby("ttm", sort=True):
            curve, source = quote_curve(quotes_at_t, spot, intrinsic)
            times.append(ttm)
            curves.append(curve)
            provenance.append(source)
        values = np.array([np.interp(MATURITIES, times, x) for x in np.array(curves).T]).T.ravel()
        flags = []
        for ttm in MATURITIES:
            hi = np.searchsorted(times, ttm)
            if hi == len(times):
                flags.append(provenance[-1] | EXTRAPOLATED)
            elif times[hi] == ttm:
                flags.append(provenance[hi])
            else:
                flags.append(provenance[hi - 1] | provenance[hi] | INTERPOLATED)
        if not days:
            A, b, counts, _ = detect(grid_t, grid_k, values)
        fixed = values + l1(A, b, values) if (A@values - b).min() < -1e-9 else values
        if not np.isfinite(fixed).all() or (A@fixed - b).min() < -1e-7:
            raise ValueError(f"Infeasible price repair on {day}")
        days.append(day)
        spots.append(spot)
        raw.append(values)
        repaired.append(fixed)
        sources.append(np.concatenate(flags))
    source = np.asarray(sources)
    np.savez_compressed(output, dates=np.array(days), spot=spots, raw=raw,
                        calls=repaired, source=source, extrapolated=(source & EXTRAPOLATED) != 0,
                        repaired=np.abs(np.asarray(repaired) - raw) > 1e-10,
                        A=A, b=b, constraint_counts=counts, grid_t=grid_t, grid_k=grid_k)
    print(f"{len(days)} surfaces, {len(grid_t)} nodes and {len(b)} constraints", flush=True)


def source_report(surface, fold):
    with np.load(surface) as data:
        years = data["dates"].astype("datetime64[Y]").astype(int) + 1970
        source = data["source"]
        repaired = data["repaired"]
        price_change = np.abs(data["calls"] - data["raw"])
        nodes = list(zip(data["grid_t"], data["grid_k"]))
    report = {}
    previous = 0
    for phase, end in zip(("train", "validation", "test"), fold):
        rows = (years > previous) & (years <= end)
        previous = end
        report[phase] = [
            dict(ttm=round(t*365), k=k, days=int(rows.sum()),
                 observed=int((source[rows, j] == 0).sum()),
                 interpolated=int(((source[rows, j] & INTERPOLATED) != 0).sum()),
                 anchor=int(((source[rows, j] & ANCHOR) != 0).sum()),
                 extrapolated=int(((source[rows, j] & EXTRAPOLATED) != 0).sum()),
                 repaired=int(repaired[rows, j].sum()),
                 mean_abs_repair=float(price_change[rows, j].mean()))
            for j, (t, k) in enumerate(nodes)
        ]
    return report
