import csv
from datetime import date
import json
from pathlib import Path
import sys

from contract_ledger import terms

def main(option_file, spot_file, output, config):
    cfg = json.loads(Path(config).read_text(encoding="utf8"))
    with open(spot_file, encoding="gb18030", newline="") as f:
        spot = {r["交易日期"]: float(r["收盘价(元)"]) for r in csv.DictReader(f)}
    selected = {}
    with open(option_file, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["contract_type"] != "CO" or not all(float(row[k] or 0)>0 for k in ("volume","position","close")):
                continue
            day, expiry = row["date"], row["expire_date"]
            year = int(day[:4])
            end = next((y for y in cfg["split_ends"] if year<=y), None)
            if end is None or int(expiry[:4])>end:
                continue
            days = (date.fromisoformat(expiry)-date.fromisoformat(day)).days
            window = next(((lo,hi) for lo,hi in cfg["windows"] if lo<=days<=hi),None)
            if window is None:
                continue
            strike, unit = terms(row,day)
            ratio = float(strike)/spot[day]
            lo, hi = cfg["moneyness"]
            if not lo<=ratio<=hi:
                continue
            term = "-".join(map(str,window))
            key = end,row["code"],term
            if key in selected and selected[key]["date"] <= day:
                continue
            lo, hi = cfg["report_moneyness"]
            selected[key] = dict(split_end=end,date=day,code=row["code"],expiry=expiry,strike=strike,
                                 unit=unit,spot=spot[day],premium=row["close"],term=term,
                                 moneyness="ITM" if ratio<lo else "OTM" if ratio>hi else "ATM")
    rows = sorted(selected.values(),key=lambda r:(r["date"],r["code"],r["term"]))
    with open(output,"w",encoding="utf8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for end in cfg["split_ends"]:
        print(end,{f"{lo}-{hi}":sum(r["split_end"]==end and r["term"]==f"{lo}-{hi}" for r in rows)
                   for lo,hi in cfg["windows"]},flush=True)

if __name__ == "__main__":
    main(*sys.argv[1:])
