"""Complete seed/group tables and paired expiry-month uncertainty, no winner selection."""
import json
from functools import lru_cache
import numpy as np
import pandas as pd
from study_core import OUT,PROJECT,CONFIG,bootstrap_legacy,write_json,digest

@lru_cache(maxsize=1)
def tail_function():
    bootstrap_legacy();from train import expected_shortfall
    return expected_shortfall

def risks(p):
    return np.sqrt(np.mean(p*p,axis=-1)),tail_function()(-p,axis=-1)

def paired_intervals(p,reference,months,block,repeats=2000):
    unique=np.unique(months);by_month=[np.flatnonzero(months==m) for m in unique]
    rng=np.random.default_rng(20261112);diff=[]
    for _ in range(repeats):
        starts=rng.integers(len(unique),size=int(np.ceil(len(unique)/block)))
        chosen=np.concatenate([(a+np.arange(block))%len(unique) for a in starts])[:len(unique)]
        ix=np.concatenate([by_month[i] for i in chosen]);a=risks(p[:,ix]);b=risks(reference[:,ix])
        diff.append([float(np.mean(x-y)) for x,y in zip(a,b)])
    low,high=np.quantile(diff,[.025,.975],axis=0)
    return dict(rmse_lower=float(low[0]),rmse_upper=float(high[0]),cvar_lower=float(low[1]),
        cvar_upper=float(high[1]),block_months=block,repeats=repeats,
        method='paired circular expiry-month blocks; seeds averaged inside each draw')

def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=OUT/'evaluation';table=pd.read_csv(root/'overall.csv')
    keys=['end','model','information','lam','cost_bp','execution']
    metrics=['rmse','cvar','mean','std','cost','turnover']
    summary=table.groupby(keys)[metrics].agg(['mean','min','max'])
    summary.columns=['_'.join(x) for x in summary.columns];summary=summary.reset_index()
    counts=table.groupby(keys).seed.nunique().rename('seed_count').reset_index()
    summary=summary.merge(counts,on=keys,validate='one_to_one')
    historical=[]
    for end in sorted(table.end.unique()):
        for family in ('C0','C1'):
            for seed in (0,1,2):
                file=PROJECT/f'工作区/数据/处理后/瓶颈验证实验/evaluation/paths/{end}_legacy_{family}_s{seed}_5bp_delay0.json'
                if not file.exists():continue
                record=json.loads(file.read_text(encoding='utf8'))
                historical.append(dict(**record['metrics'],source_sha256=digest(file),
                    comparison='historical performance reference; preprocessing/initialization/information differ'))
    pd.DataFrame(historical).to_csv(root/'historical_references.csv',index=False)
    main_old=[x for x in historical if x['end']==2025 and x['kind']=='legacy_C1']
    if len(main_old)==3:
        for metric in ('rmse','cvar'):
            reference=float(np.mean([r[metric] for r in main_old]))
            summary[f'{metric}_minus_old_C1']=np.where((summary.end==2025)&(summary.cost_bp==5)&
                summary.execution.eq('same_close'),summary[f'{metric}_mean']-reference,np.nan)
    summary.to_csv(root/'seed_summary.csv',index=False)
    groups=[];cache={};intervals=[]
    for _,row in table.iterrows():
        key=f"{row['key']}_cost{int(row.cost_bp)}_{row.execution}"
        d=pd.read_csv(root/f'{key}.paths.csv',dtype={'episode':str})
        cache[(int(row.end),row.model,row.information,float(row.lam),int(row.cost_bp),row.execution,int(row.seed))]=d
        for name in ('term','moneyness','month','cross_dividend','adjusted'):
            for label,g in d.groupby(name):
                rmse,cvar=risks(g.pnl.to_numpy())
                groups.append(dict(key=row['key'],end=int(row.end),model=row.model,seed=int(row.seed),
                    information=row.information,lam=float(row.lam),cost_bp=int(row.cost_bp),execution=row.execution,
                    group=name,label=label,n=len(g),rmse=float(rmse),cvar=float(cvar),
                    mean=float(g.pnl.mean()),cost=float(g.cost.mean()),turnover=float(g.turnover.mean())))
    pd.DataFrame(groups).to_csv(root/'group_metrics.csv',index=False)
    for _,r in summary.iterrows():
        if r.model=='nsde':continue
        prefix=(int(r.end),r.model,r.information,float(r.lam),int(r.cost_bp),r.execution)
        base=(int(r.end),'nsde',r.information,float(r.lam),int(r.cost_bp),r.execution)
        if not all((*prefix,s) in cache and (*base,s) in cache for s in (0,1,2)):continue
        first=cache[(*prefix,0)]
        for s in (0,1,2):
            if not np.array_equal(first.episode,cache[(*prefix,s)].episode) or not np.array_equal(first.episode,cache[(*base,s)].episode):
                raise ValueError('Paired path ID mismatch')
        p=np.stack([cache[(*prefix,s)].pnl for s in (0,1,2)])
        ref=np.stack([cache[(*base,s)].pnl for s in (0,1,2)])
        for block in (1,3,6):
            intervals.append(dict(end=int(r.end),model=r.model,information=r.information,
                lam=float(r.lam),cost_bp=int(r.cost_bp),execution=r.execution,
                **paired_intervals(p,ref,first.month.to_numpy(),block,CONFIG['bootstrap'])))
    pd.DataFrame(intervals).to_csv(root/'paired_intervals.csv',index=False)
    for end in sorted(table.end.unique()):
        main=summary[(summary.end==end)&(summary.cost_bp==5)&summary.execution.eq('same_close')]
        fig,ax=plt.subplots(figsize=(10,7))
        for _,r in main.iterrows():
            ax.scatter(100*r.rmse_mean,100*r.cvar_mean)
            ax.annotate(f'{r.model}/{r.information}/lambda={r.lam:g}',(100*r.rmse_mean,100*r.cvar_mean),fontsize=6)
        ax.set(xlabel='RMSE (% initial notional)',ylabel='CVaR95 loss (% initial notional)',title=f'{end}: error-tail trade-offs, 5bp')
        ax.grid(alpha=.2);fig.tight_layout();fig.savefig(root/f'tradeoffs_{end}.png',dpi=180);plt.close(fig)
    write_json(root/'report_status.json',dict(tables_complete=True,seed_rows=len(summary),group_rows=len(groups),
        interval_rows=len(intervals),exploratory=True,multiple_comparison_adjustment=False,
        limitations=['historically observed evaluation sets','24 main expiry months','training-internal forecast diagnostics']))

if __name__=='__main__':run()
