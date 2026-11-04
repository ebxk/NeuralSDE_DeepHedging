import sys

import numpy as np
import pandas as pd

from contract_ledger import terms

def main(option_file, spot_file, upstream, output):
    sys.path.insert(0, upstream)
    from arbitragerepair.constraints import detect
    from arbitragerepair.repair import l1
    from cvxopt import solvers
    solvers.options["glpk"] = {"msg_lev":"GLP_MSG_OFF"}

    columns = ["date", "code", "trading_code", "list_date", "expire_date",
               "contract_type", "close", "volume", "position"]
    options = pd.read_csv(option_file, usecols=columns)
    spot = pd.read_csv(spot_file, encoding="gb18030").set_index("交易日期")["收盘价(元)"]
    options = options[(options.contract_type == "CO") & (options.close > 0)
                      & (options.volume > 0) & (options.position > 0)].copy()
    options["strike"] = [float(terms(row, row["date"])[0])
                         for row in options.to_dict("records")]
    options["ttm"] = (pd.to_datetime(options.expire_date) -
                      pd.to_datetime(options.date)).dt.days
    strikes = np.array([.90, .95, 1., 1.05, 1.10])
    maturities = np.array([14, 30, 60, 90])
    grid_t = np.repeat(maturities/365, len(strikes))
    grid_k = np.tile(strikes, len(maturities))
    intrinsic = np.maximum(1-strikes, 0)
    days, spots, raw, repaired = [], [], [], []
    extrapolated = []
    for day, quotes in options.groupby("date"):
        price = spot[day]
        curves, times = [intrinsic], [0]
        strike_out = [np.zeros(len(strikes), dtype=bool)]
        for ttm, slice_ in quotes[quotes.ttm > 0].groupby("ttm"):
            # Prefer the most active quote for duplicate effective strikes.
            slice_ = slice_.sort_values(["volume", "code"], ascending=[False, True])
            slice_ = slice_.drop_duplicates("strike").sort_values("strike")
            k = slice_.strike.to_numpy()/price
            c = slice_.close.to_numpy()/price
            # Zero-strike anchor and linear extrapolation bounded by intrinsic.
            strike_out.append((strikes < k[0]) | (strikes > k[-1]))
            slope = np.clip((c[-1]-c[-2])/(k[-1]-k[-2]), -1, 0) if len(k)>1 else np.clip((c[-1]-1)/k[-1],-1,0)
            k, c = np.r_[0., k], np.r_[1., c]
            curve = np.interp(strikes, k, c)
            outside = strikes > k[-1]
            curve[outside] = np.maximum(intrinsic[outside], c[-1]+(strikes[outside]-k[-1])*slope)
            times.append(ttm)
            curves.append(curve)
        if len(times) == 1:
            continue
        values = np.array([np.interp(maturities, times, x) for x in np.array(curves).T]).T.ravel()
        for expiry in maturities:
            hi = min(np.searchsorted(times, expiry), len(times)-1)
            lo = hi if times[hi] <= expiry else hi-1
            extrapolated.extend(strike_out[lo] | strike_out[hi] | (expiry > times[-1]))
        if not days:
            A, b, *_ = detect(grid_t, grid_k, values)
        fixed = values + l1(A, b, values) if (A@values-b).min() < -1e-9 else values
        if (A@fixed-b).min() < -1e-7:
            raise ValueError(f"Repair infeasible on {day}")
        days.append(day)
        spots.append(price)
        raw.append(values)
        repaired.append(fixed)
    np.savez_compressed(output, dates=np.array(days), spot=spots, raw=raw,
                        calls=repaired, extrapolated=np.array(extrapolated).reshape(len(days),-1), A=A, b=b, grid_t=grid_t, grid_k=grid_k)
    print(f"{len(days)} surfaces; mean absolute normalized repair {np.mean(np.abs(np.array(raw)-repaired)):.6g}")

if __name__ == "__main__":
    main(*sys.argv[1:])
