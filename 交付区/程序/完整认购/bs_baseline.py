import csv
from datetime import date
from itertools import groupby
from math import erf, log, sqrt
from pathlib import Path
import sys

from contract_ledger import ledger

def cdf(x):
    return (1 + erf(x / sqrt(2))) / 2

def delta(row, previous_iv):
    spot, strike, quote = map(float, (row["spot"], row["strike"], row["option_close"]))
    tau = (date.fromisoformat(row["expiry"]) - date.fromisoformat(row["date"])).days / 365
    m = log(spot / strike)
    if max(spot - strike, 0) < quote < spot:
        low, high = 0.0, 12.0
        for _ in range(50):
            v = (low + high) / 2
            d1 = m / v + v / 2
            if spot * cdf(d1) - strike * cdf(d1-v) < quote:
                low = v
            else:
                high = v
        iv = (low + high) / (2 * sqrt(tau))
        fallback = False
    else:
        iv = previous_iv
        fallback = True
    v = iv * sqrt(tau)
    return cdf(m/v + v/2), iv, fallback

def main(source, destination, config):
    import json
    cfg = json.loads(Path(config).read_text(encoding="utf8"))
    with Path(source).open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    output = []
    fallbacks = 0
    for episode, group in groupby(rows, key=lambda row: row["episode"]):
        path = list(group)
        if int(path[0]["split_end"]) != cfg["split_ends"][-1]:
            continue
        ratios = []
        iv = .25
        for row in path[:-1]:
            ratio, iv, fallback = delta(row, iv)
            ratios.append(ratio)
            fallbacks += fallback
        for bp in cfg["costs_bp"]:
            for row in ledger(path, ratios, float(path[0]["option_close"]), bp/10000):
                output.append(dict(episode=episode, cost_bp=bp, expiry=path[0]["expiry"],
                                   initial_notional=float(path[0]["spot"])*int(path[0]["unit"]), **row))
    with Path(destination).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print("Fallback decisions:", fallbacks)
    for bp in cfg["costs_bp"]:
        pnl = [r["wealth"]/r["initial_notional"] for r in output
               if r["cost_bp"] == bp and r["date"] == r["expiry"]]
        print(bp, len(pnl), "RMSE", sqrt(sum(x*x for x in pnl)/len(pnl)))

if __name__ == "__main__":
    main(*sys.argv[1:])
