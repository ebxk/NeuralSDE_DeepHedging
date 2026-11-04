import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from contract_ledger import EVENTS

def main(data_dir,result_dir,config):
    data,result = Path(data_dir),Path(result_dir)
    cfg=json.loads(Path(config).read_text(encoding="utf8"))
    episodes=pd.read_csv(data/"episodes.csv",dtype={"code":str})
    episodes.index.name="episode"
    test=episodes[episodes.split_end==cfg["split_ends"][-1]].copy()
    test["cross_dividend"]=False
    for effective, record, _, _, _ in EVENTS:
        test["cross_dividend"] |= (test.date<=record) & (test.expiry>=effective)
    days=pd.read_csv(data/"episode_days.csv",usecols=["episode","date","spot"])
    price=days.set_index(["episode","date"]).spot.to_dict()
    summaries={}
    for file in ("bs_ledger.csv","dh_ledger.csv"):
        with (data/file).open(encoding="utf8",newline="") as f:
            for row in csv.DictReader(f):
                eid=int(row["episode"])
                group=row.get("group","BS Delta")
                bp=int(row.get("bp") or row["cost_bp"])
                seed=row.get("seed","")
                key=group,bp,seed,eid
                entry=summaries.setdefault(key,dict(group=group,bp=bp,seed=seed,episode=eid,
                    cost=0.,turnover=0.,pnl=None))
                notional=float(row["initial_notional"])
                entry["cost"]+=float(row["cost"])/notional
                entry["turnover"]+=abs(float(row["traded"]))*price[eid,row["date"]]/notional
                if row["date"]==row["expiry"]:
                    assert entry["pnl"] is None and float(row["shares"])==0
                    entry["pnl"]=float(row["wealth"])/notional
    frame=pd.DataFrame(summaries.values()).join(test[["code","expiry","term","moneyness","cross_dividend"]],on="episode",validate="many_to_one")
    assert np.isfinite(frame[["pnl","cost","turnover"]].to_numpy()).all()
    np.testing.assert_allclose(frame.cost,frame.turnover*frame.bp/10000,atol=1e-8)
    frame.to_csv(result/"episode_metrics.csv",index=False)
    rows=[]
    assert frame.groupby(["group","bp","seed"]).ngroups==len(cfg["costs_bp"])*(1+3*len(cfg["seeds"]))
    for (name,bp,seed),g in frame.groupby(["group","bp","seed"],sort=False):
        assert len(g)==len(test) and set(g.episode)==set(test.index)
        slices=[("overall","all",g)]
        for col in ("term","moneyness","cross_dividend"):
            slices.extend((col,str(label),part) for label,part in g.groupby(col))
        slices.extend(("term_moneyness",f"{term} {money}",part)
                      for (term,money),part in g.groupby(["term","moneyness"]))
        for axis,label,part in slices:
            loss=np.sort(-part.pnl.to_numpy())[::-1]
            tail=.05*len(loss)
            n=int(tail)
            rows.append(dict(group=name,bp=bp,seed=seed,axis=axis,subset=label,
                n=len(loss),contracts=part.code.nunique(),expiries=part.expiry.nunique(),
                rmse=np.sqrt(np.mean(loss**2)),cvar95=(loss[:n].sum()+(tail-n)*loss[n])/tail,
                mean_pnl=part.pnl.mean(),cost=part.cost.mean(),turnover=part.turnover.mean()))
    per_seed=pd.DataFrame(rows)
    per_seed.to_csv(result/"metrics_by_seed.csv",index=False)
    summary=per_seed.groupby(["group","bp","axis","subset"],sort=False).agg(
        n=("n","first"),contracts=("contracts","first"),expiries=("expiries","first"),
        rmse=("rmse","mean"),rmse_min=("rmse","min"),rmse_max=("rmse","max"),
        cvar95=("cvar95","mean"),cvar95_min=("cvar95","min"),cvar95_max=("cvar95","max"),
        mean_pnl=("mean_pnl","mean"),cost=("cost","mean"),turnover=("turnover","mean")).reset_index()
    summary.to_csv(result/"summary.csv",index=False)
    s=np.load(data/"surfaces.npz")
    factors=np.load(data/"factors.npz")
    assert np.array_equal(s["dates"],factors["dates"])
    pd.DataFrame(dict(date=s["dates"],extrapolated_share=s["extrapolated"].mean(axis=1),
        mean_abs_repair=np.abs(s["calls"]-s["raw"]).mean(axis=1),
        factor_rmse=factors["reconstruction_rmse"])).to_csv(result/"curve_diagnostics.csv",index=False)
    overall=summary[(summary.bp==5)&(summary.axis=="overall")]
    print(overall[["group","n","rmse","cvar95","mean_pnl","cost"]].to_string(index=False),flush=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    names=["BS Delta","GBM DH","Neural SDE DH","Neural SDE DH factors"]
    for name in names:
        pnl=frame[(frame.group==name)&(frame.bp==5)].pnl*100
        axes[0].step(np.sort(pnl),np.arange(1,len(pnl)+1)/len(pnl),where="post",label=name)
    axes[0].set(xlabel="Terminal P&L (% of initial notional)",ylabel="Empirical CDF")
    axes[0].legend(fontsize=8)
    chart=overall.set_index("group").loc[names]
    ix=np.arange(len(chart))
    axes[1].bar(ix-.18,100*chart.rmse,.36,label="RMSE")
    axes[1].bar(ix+.18,100*chart.cvar95,.36,label="CVaR95")
    axes[1].set_xticks(ix,["BS Delta","GBM DH","SDE DH","SDE DH + factors"],rotation=20)
    axes[1].set_ylabel("% of initial notional")
    axes[1].legend()
    fig.savefig(result/"comparison_5bp.png",dpi=160)

if __name__=="__main__":
    main(*sys.argv[1:])
