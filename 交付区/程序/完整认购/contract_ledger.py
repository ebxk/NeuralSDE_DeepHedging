from bisect import bisect_left, bisect_right
import csv
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import sys

# SSE effective date and unit; ChinaAMC record date, payment date and cash/share.
EVENTS = [
    ("2016-11-29","2016-11-28","2016-12-02","0.053",10220),
    ("2017-11-28","2017-11-27","2017-12-01","0.054",10185),
    ("2018-12-03","2018-11-30","2018-12-06","0.049",10202),
    ("2019-12-02","2019-11-29","2019-12-05","0.047",10163),
    ("2020-11-30","2020-11-27","2020-12-03","0.051",10145),
    ("2021-11-29","2021-11-26","2021-12-02","0.041",10129),
    ("2022-12-01","2022-11-30","2022-12-06","0.037",10142),
    ("2023-11-27","2023-11-24","2023-11-30","0.039",10161),
    ("2024-12-02","2024-11-29","2024-12-05","0.055",10205),
    ("2025-12-17","2025-12-16","2025-12-22","0.080",10265),
]

def terms(row, day):
    strike = Decimal(row["trading_code"][-5:]) / 1000
    unit = 10000
    for effective, _, _, _, new_unit in EVENTS:
        if row["list_date"] < effective <= min(day, row["expire_date"]):
            strike = (strike * unit / new_unit).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
            unit = new_unit
    return strike, unit

def ledger(rows, ratios, premium, cost_rate, events=EVENTS):
    """Return one ledger row per close; last row settles and liquidates."""
    ratios = list(ratios)
    if len(ratios) != len(rows)-1 or any(not 0 <= x <= 1 for x in ratios):
        raise ValueError("One hedge ratio in [0,1] per pre-expiry close is required")
    cash, shares, receivable = premium * int(rows[0]["unit"]), 0.0, 0.0
    entitlements = {}
    result = []
    for i, row in enumerate(rows):
        day, spot = row["date"], float(row["spot"])
        unit, strike = int(row["unit"]), float(row["strike"])
        accrued = paid = 0.0
        for effective, _, payment, _, _ in events:
            if effective == day:
                accrued += entitlements.get(effective, 0.0)
            if payment == day:
                paid += entitlements.get(effective, 0.0)
        receivable += accrued - paid
        cash += paid
        final = i == len(rows)-1
        target = 0.0 if final else ratios[i] * unit
        traded = target - shares
        cost = abs(traded) * spot * cost_rate
        liability = unit * max(spot-strike, 0) if final else 0.0
        cash -= traded * spot + cost + liability
        shares = target
        for effective, record, _, amount, _ in events:
            if record == day:
                entitlements[effective] = shares * float(amount)
        result.append(dict(date=day, shares=shares, traded=traded, cost=cost,
                           accrued=accrued, paid=paid, receivable=receivable,
                           liability=liability, cash=cash, wealth=cash+receivable+shares*spot))
    return result

def main(option_file, spot_file, episodes_file, destination):
    with open(spot_file, encoding="gb18030", newline="") as f:
        spot = {r["交易日期"]: float(r["收盘价(元)"]) for r in csv.DictReader(f)}
    with open(episodes_file, encoding="utf-8", newline="") as f:
        episodes = list(csv.DictReader(f))
    quotes, contracts = {}, {}
    with open(option_file, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = row["code"], row["date"]
            if key in quotes:
                raise ValueError(f"Duplicate quote {key}")
            quotes[key] = row["close"]
            contracts[row["code"]] = row
    adjusted = 0
    for row in contracts.values():
        strike, unit = terms(row, row["expire_date"])
        assert (strike, unit) == (Decimal(row["exercise_price"]), int(row["contract_unit"])), row["code"]
        adjusted += unit != 10000
    fields = ["episode", "split_end", "term", "moneyness", "date", "code", "expiry", "strike", "unit", "spot", "option_close"]
    days = sorted(spot)
    count = 0
    with open(destination, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, ep in enumerate(episodes):
            path = []
            for day in days[bisect_left(days, ep["date"]):bisect_right(days, ep["expiry"])]:
                price = float(quotes[ep["code"], day])
                assert price > 0 and spot[day] > 0
                strike, unit = terms(contracts[ep["code"]], day)
                path.append(dict(episode=i, split_end=ep["split_end"], term=ep["term"], moneyness=ep["moneyness"], date=day, code=ep["code"],
                                 expiry=ep["expiry"], strike=strike, unit=unit, spot=spot[day],
                                 option_close=price))
            assert path[0]["date"] == ep["date"] and path[-1]["date"] == ep["expiry"]
            assert (float(path[0]["strike"]),path[0]["unit"]) == (float(ep["strike"]),int(ep["unit"]))
            writer.writerows(path)
            count += len(path)
    print(f"Validated {len(contracts)} contracts ({adjusted} adjusted); {count} daily rows.")

if __name__ == "__main__":
    main(*sys.argv[1:])
