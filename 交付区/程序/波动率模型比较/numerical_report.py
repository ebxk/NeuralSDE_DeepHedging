"""Aggregate recorded numerical approximations and measured compute costs."""
import json
import math
import statistics
from collections import Counter
import pandas as pd
from study_core import OUT,HERE,PROJECT,CONFIG,digest,write_json

FIGURES=tuple(f'{kind}_l{lam:g}.png' for kind in ('year_stability','pressure_main') for lam in (0.,.1))

def comparison_figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    root=OUT/'evaluation';table=pd.read_csv(root/'seed_summary.csv')
    overall=pd.read_csv(root/'overall.csv');sample_counts=overall.groupby('end').n.first().to_dict()
    styles={'nsde':('#252b38','*','Neural SDE'),'gbm':('#2f6b9a','o','GBM'),
        'original_linear':('#8db1c7','s','Original / low-capacity'),'svi_linear':('#c17a2c','^','SVI / low-capacity'),
        'ssvi_linear':('#d0ad50','v','SSVI / low-capacity'),'essvi_linear':('#ad6487','D','eSSVI / low-capacity'),
        'heston':('#586c3b','P','Heston'),'sabr':('#8aa45d','X','SABR')}
    models=[m for m in styles if m in set(table.model)];offsets=np.linspace(-.30,.30,len(models))
    for kind in ('year_stability','pressure_main'):
        for lam in (0.,.1):
            fig,axes=plt.subplots(2,2,figsize=(13,8),sharey='col')
            upper_limits=[0.,0.]
            for row,info in enumerate(('base','surface')):
                subset=table[table.information.eq(info)&table.lam.eq(lam)]
                if kind=='year_stability':
                    subset=subset[(subset.cost_bp==5)&subset.execution.eq('same_close')]
                    keys=[2021,2022,2023,2025];labels=['2021','2022','2023','2024-2025']
                    selector=lambda g,k:g[g.end==k]
                else:
                    subset=subset[subset.end==2025];keys=[(0,'same_close'),(5,'same_close'),(10,'same_close'),(5,'next_close')]
                    labels=['0bp\nsame close','5bp\nsame close','10bp\nsame close','5bp\nnext close']
                    selector=lambda g,k:g[(g.cost_bp==k[0])&g.execution.eq(k[1])]
                for col,metric in enumerate(('rmse','cvar')):
                    ax=axes[row,col]
                    for offset,model in zip(offsets,models):
                        group=subset[subset.model==model];parts=[selector(group,k) for k in keys]
                        if any(len(g)!=1 or int(g.seed_count.iloc[0])!=3 for g in parts):
                            raise ValueError('Incomplete three-seed figure series')
                        mean=np.array([g[f'{metric}_mean'].iloc[0] for g in parts])*100
                        lo=np.array([g[f'{metric}_min'].iloc[0] for g in parts])*100
                        hi=np.array([g[f'{metric}_max'].iloc[0] for g in parts])*100
                        upper_limits[col]=max(upper_limits[col],float(hi.max()))
                        color,marker,_=styles[model]
                        ax.errorbar(np.arange(4)+offset,mean,yerr=np.maximum(0,np.array([mean-lo,hi-mean])),
                            fmt=marker,color=color,markersize=5,capsize=2,elinewidth=.8,linestyle='none')
                    ax.set_xticks(np.arange(4),labels);ax.set_xlim(-.55,3.55)
                    ax.set_title(f'{"Base inputs" if info=="base" else "Shared surface inputs"}: {metric.upper() if metric=="rmse" else "CVaR95 loss"}',fontsize=11)
                    ax.set_ylabel('% of initial notional');ax.grid(axis='y',alpha=.2)
                    ax.spines[['top','right']].set_visible(False)
            for col,maximum in enumerate(upper_limits):axes[0,col].set_ylim(0,maximum*1.08)
            handles=[Line2D([],[],color=c,marker=m,linestyle='none',label=label) for c,m,label in (styles[k] for k in models)]
            fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.055),ncol=4,frameon=False,fontsize=9)
            fig.suptitle(f'{"Cross-year comparison: 5bp, same close" if kind=="year_stability" else "Cost and execution pressure: 2024-2025"}; lambda={lam:g}',fontsize=14)
            visible_counts=sample_counts if kind=='year_stability' else {2025:sample_counts[2025]}
            samples='; '.join(f'{"2024-2025" if int(year)==2025 else year}: n={n}' for year,n in sorted(visible_counts.items()))
            fig.text(.5,.012,'Dots: three-seed means. Whiskers: seed ranges, not confidence intervals.\nObserved historical evaluation. '+samples,ha='center',fontsize=9)
            fig.tight_layout(rect=(0,.17,1,.94));fig.savefig(root/f'{kind}_l{lam:g}.png',dpi=160);plt.close(fig)
    write_json(root/'comparison_figures.json',dict(complete=True,source=digest(HERE/'numerical_report.py'),
        parents={str(root/name):digest(root/name) for name in ('seed_summary.csv','overall.csv')},sample_counts=sample_counts,
        output_sha256={name:digest(root/name) for name in FIGURES},
        range_meaning='three policy seeds, not confidence intervals',metric_units='percent of initial notional'))

def surface_cost(records,end):
    selected=[r for r in records if r['day']<str(end+1)]
    elapsed=[r['seconds'] for r in selected if r['seconds'] is not None and math.isfinite(r['seconds'])]
    if not selected or not elapsed:raise ValueError('No recorded surface calibration timing')
    return dict(dates=len(selected),qualified_dates=sum(r['status']=='complete' for r in selected),
        rejected_dates=sum(r['status']=='numerical_rejected' for r in selected),timed_dates=len(elapsed),
        sum_fit_seconds=sum(elapsed),median_fit_seconds=statistics.median(elapsed))


def scalar_diagnostics(value,prefix=''):
    if isinstance(value,dict):
        for key,item in value.items():
            if key in ('tables','origin_fallbacks','parents','sources','identity'):continue
            yield from scalar_diagnostics(item,f'{prefix}.{key}' if prefix else key)
    elif isinstance(value,(list,tuple)):
        for i,item in enumerate(value):yield from scalar_diagnostics(item,f'{prefix}[{i}]')
    elif isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value):
        yield prefix,float(value)


def run():
    root=OUT/'evaluation';root.mkdir(exist_ok=True);rows=[];parents={};coverage=[]
    def read(path):
        parents[str(path)]=digest(path);return json.loads(path.read_text(encoding='utf8'))
    for family in ('svi','ssvi','essvi'):
        for view in ('full','heldout'):
            records=[]
            for path in sorted((OUT/'surfaces'/family).glob(f'*/{view}.json')):
                r=read(path);records.append(dict(day=path.parent.name,status=r['status'],
                    seconds=r.get('model',{}).get('diagnostics',{}).get('seconds')))
            for _,_,end in CONFIG['folds']:
                for key,value in surface_cost(records,end).items():
                    rows.append(dict(end=end,model=family+'_native',stage='daily_'+view+'_surface_calibration',diagnostic=key,value=value))
    for _,_,end in CONFIG['folds']:
        for model in CONFIG['models']:
            ready=OUT/f'inputs/{end}/sim_{model}.ready.json'
            if ready.exists():
                r=read(ready);coverage.append(dict(end=end,model=model,status='complete'))
                for key,value in scalar_diagnostics({'seconds':r.get('seconds'),'simulation':r.get('diagnostics',{})}):
                    rows.append(dict(end=end,model=model,stage='export',diagnostic=key,value=value))
                tables=r.get('diagnostics',{}).get('tables',{})
                values={}
                for table in tables.values():
                    for key,value in scalar_diagnostics(table.get('diagnostics',{})):values.setdefault(key,[]).append(value)
                for key,observations in values.items():
                    for suffix,value in (('min',min(observations)),('max',max(observations)),('mean',sum(observations)/len(observations))):
                        rows.append(dict(end=end,model=model,stage='origin_table_aggregate',diagnostic=f'{key}.{suffix}',value=value))
            else:
                failure=OUT/f'failures/{end}_{model}_base_l0_s0.json'
                r=read(failure)
                if r.get('status')!='blocked_by_generator' or r.get('category')!='numerical_rejection':
                    raise ValueError('Unresolved generator has no scientific numerical rejection')
                coverage.append(dict(end=end,model=model,status='declared_failure'))
            market=OUT/f'markets/{end}/{model}/market.json'
            if market.exists():
                r=read(market)
                for key,value in scalar_diagnostics({k:r[k] for k in ('diagnostics2','diagnostics4','numeric',
                    'full_grid_minimum_slack','paired_terminal_state_rmse','seconds','fit_seconds','validation_nll') if k in r}):
                    rows.append(dict(end=end,model=model,stage='fit_and_simulate',diagnostic=key,value=value))
            global_fit=OUT/f'markets/{end}/{model}/global.json'
            if global_fit.exists():
                r=read(global_fit)
                seconds=r['model'].get('diagnostics',{}).get('seconds')
                if seconds is not None:rows.append(dict(end=end,model=model,stage='global_Q_fit_only',diagnostic='seconds',value=seconds))
        path=OUT/f'direct/{end}/path_pnl.csv'
        if path.exists():
            parents[str(path)]=digest(path);frame=pd.read_csv(path)
            episode_path=PROJECT/f'工作区/数据/处理后/创新组合实验/{end}/episode_days.csv'
            parents[str(episode_path)]=digest(episode_path);episodes=pd.read_csv(episode_path)
            lengths=episodes[episodes.split_end==end].groupby('episode').size()
            decisions=int((lengths-1).sum())
            for (family,delta),g in frame[(frame.cost_bp==5)&frame.execution.eq('same_close')].groupby(['model','delta']):
                if len(g)!=len(lengths) or set(g.episode)!=set(lengths.index):raise ValueError('Direct diagnostic path denominator mismatch')
                for key,value in [('fallback_fraction',g.fallback_decisions.sum()/decisions),
                                  ('decision_clip_fraction',g.decision_clip_count.sum()/decisions)]:
                    rows.append(dict(end=end,model=family+'_'+delta,stage='direct_5bp_path_weighted_decisions',diagnostic=key,value=float(value)))
            ledger=path.with_name('daily_ledger.csv');parents[str(ledger)]=digest(ledger);counts=Counter()
            for chunk in pd.read_csv(ledger,usecols=['model','delta','cost_bp','execution','raw_decision','fallback_source'],chunksize=100000):
                used=chunk[(chunk.cost_bp==5)&chunk.execution.eq('same_close')&chunk.raw_decision.notna()]
                for key,count in used.groupby(['model','delta','fallback_source']).size().items():counts[key]+=int(count)
            for family,delta in frame.groupby(['model','delta']).groups:
                if sum(count for key,count in counts.items() if key[:2]==(family,delta))!=decisions:
                    raise ValueError('Direct source fractions do not cover every path decision')
            for (family,delta,source),count in counts.items():
                rows.append(dict(end=end,model=family+'_'+delta,stage='direct_5bp_path_weighted_decisions',
                    diagnostic='source_fraction.'+source,value=count/decisions))
    pd.DataFrame(rows).to_csv(root/'numerical_diagnostics.csv',index=False)
    frame=pd.read_csv(OUT/'evaluation/overall.csv');parents[str(OUT/'evaluation/overall.csv')]=digest(OUT/'evaluation/overall.csv')
    columns=['end','model','information','lam','seed','cost_bp','execution','n',
        'inference_parameters','training_seconds','replay_seconds']
    frame[columns].to_csv(root/'compute_costs.csv',index=False)
    comparison_figures()
    result=dict(complete=True,coverage=coverage,rows=len(rows),source=digest(HERE/'numerical_report.py'),
        parents=parents,output_sha256={name:digest(root/name) for name in ('numerical_diagnostics.csv','compute_costs.csv')},
        interpretation='recorded numerical diagnostics; paired smaller-step samples do not prove policy invariance',
        timing='measured CPU wall time; overlapping jobs share resources; training time repeated across pressure cases; native curve costs are shared-date prefixes across folds, not independent refits; global Q timing excludes daily filtering and export; do not sum duplicate shared work')
    write_json(root/'numerical_status.json',result);return result

if __name__=='__main__':run()
