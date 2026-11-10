"""Saved-result summaries and common expiry-month block intervals."""
import bootstrap
import csv
import json
from pathlib import Path
import shutil
import numpy as np
from prepare_common import write_json

def weighted_metrics(pnl,w):
    denominator=w.sum(1)
    rmse=[];tail=[]
    for p in pnl:
        rmse.append(np.sqrt((w*p[None,:]**2).sum(1)/denominator))
        order=np.argsort(-p)[::-1];loss=-p[order];weights=w[:,order]
        prior=np.cumsum(weights,axis=1)-weights
        used=np.minimum(weights,np.maximum(.05*denominator[:,None]-prior,0))
        tail.append((used*loss).sum(1)/(.05*denominator))
    return np.mean(rmse,axis=0),np.mean(tail,axis=0)

def finish(project,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from reference_replay import run
    run(project,out)
    records=[json.loads(p.read_text(encoding='utf8')) for p in (out/'evaluation/paths').glob('*.json')]
    metrics=[r['metrics'] for r in records]
    if not metrics:return dict(status='incomplete',reason='No verified historical ledgers')
    with (out/'evaluation/metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in metrics for k in row)))
        writer.writeheader();writer.writerows(metrics)
    groups={}
    subgroup=[]
    for r in records:
        m=r['metrics'];key=(m['end'],m['cost_bp'],m['delay'],m['kind'])
        groups.setdefault(key,[]).append(r)
        for column in ('term','moneyness','cross_dividend','adjusted','expiry'):
            for value in sorted({str(x[column]) for x in r['rows']}):
                selected=[x for x in r['rows'] if str(x[column])==value]
                pnl=np.array([x['pnl'] for x in selected]);loss=np.sort(-pnl)[::-1]
                mass=.05*len(loss);weight=np.clip(mass-np.arange(len(loss)),0,1)
                subgroup.append(dict(**{k:m[k] for k in ('end','kind','seed','cost_bp','delay')},
                    dimension=column,value=value,n=len(pnl),rmse=float(np.sqrt(np.mean(pnl*pnl))),
                    cvar=float((loss*weight).sum()/mass),mean=float(pnl.mean()),std=float(pnl.std()),
                    cost=float(np.mean([x['cost'] for x in selected])),
                    turnover=float(np.mean([x['turnover'] for x in selected]))))
    write_json(out/'evaluation/subgroups.json',subgroup)
    aggregate=[]
    for key,items in groups.items():
        row=dict(end=key[0],cost_bp=key[1],delay=key[2],kind=key[3],seeds=len(items))
        for field in ('rmse','cvar','mean','std','cost','turnover','parameters','training_seconds','replay_seconds'):
            values=[r['metrics'][field] for r in items if field in r['metrics']]
            if values:row[field]=dict(mean=float(np.mean(values)),low=float(min(values)),high=float(max(values)))
        aggregate.append(row)
    write_json(out/'evaluation/seed_summary.json',aggregate)
    intervals=[]
    for end,cost,delay in sorted({k[:3] for k in groups}):
        base=sorted(groups.get((end,cost,delay,'plain'),[]),key=lambda r:r['metrics']['seed'])
        if len(base)!=3:continue
        ids=[r['episode'] for r in base[0]['rows']]
        months=np.array([r['expiry'][:7] for r in base[0]['rows']]);levels=np.unique(months)
        baseline=np.array([[r['pnl'] for r in item['rows']] for item in base])
        for length in (1,3,6):
            rng=np.random.default_rng(20261110+int(end)+length)
            counts=np.zeros((2000,len(levels)))
            for b in range(2000):
                start=rng.integers(len(levels),size=int(np.ceil(len(levels)/length)))
                sample=(start[:,None]+np.arange(length)).ravel()[:len(levels)]%len(levels)
                counts[b]=np.bincount(sample,minlength=len(levels))
            w=counts[:,np.searchsorted(levels,months)]
            a=weighted_metrics(baseline,w)
            references={'plain':a}
            old=groups.get((end,cost,delay,'legacy_C1'),[])
            if len(old)==3:
                old=sorted(old,key=lambda r:r['metrics']['seed'])
                assert all([x['episode'] for x in r['rows']]==ids for r in old)
                references['legacy_C1']=weighted_metrics(np.array([[x['pnl'] for x in r['rows']] for r in old]),w)
            for key,items in groups.items():
                if key[:3]!=(end,cost,delay):continue
                if len(items)!=3 and not (len(items)==1 and items[0]['metrics']['seed']==-1):continue
                items=sorted(items,key=lambda r:r['metrics']['seed'])
                assert all([r['episode'] for r in item['rows']]==ids for item in items)
                b=weighted_metrics(np.array([[r['pnl'] for r in item['rows']] for item in items]),w)
                for reference_name,reference_values in references.items():
                    if key[3]==reference_name:continue
                    for metric,x,y in zip(('rmse','cvar'),reference_values,b):
                        delta=y-x
                        intervals.append(dict(end=end,cost_bp=cost,delay=delay,kind=key[3],metric=metric,
                            reference=reference_name,block_months=length,
                            low=float(np.quantile(delta,.025)),high=float(np.quantile(delta,.975)),
                            evidence='limited' if np.quantile(delta,.025)<=0<=np.quantile(delta,.975) else 'directional',
                            repetitions=2000,adjusted_for_multiple_comparisons=False))
    write_json(out/'evaluation/paired_intervals.json',intervals)
    fig,ax=plt.subplots(figsize=(7,4))
    for key,items in groups.items():
        if key[:3]!=(2025,5,False):continue
        ax.scatter(np.mean([r['metrics']['rmse'] for r in items])*100,
                   np.mean([r['metrics']['cvar'] for r in items])*100,label=key[3])
    ax.set_xlabel('RMSE (% initial notional)');ax.set_ylabel('CVaR95 loss (%)');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'evaluation/risk_tradeoff.png',dpi=160);plt.close(fig)
    figures=[('rolling_stability.png','year'),('cost_execution_stress.png','cost')]
    for filename,dimension in figures:
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        names=sorted({k[3] for k in groups if k[3] not in {'BS','CurveDelta','LocalMV','legacy_GBM'}})
        for name in names:
            for delayed in ((False,) if dimension=='year' else (False,True)):
                points=[]
                for key,items in groups.items():
                    if key[3]!=name or key[2]!=delayed:continue
                    if dimension=='year' and key[1]!=5:continue
                    if dimension=='cost' and key[0]!=2025:continue
                    points.append((key[0] if dimension=='year' else key[1],
                                   *[100*np.mean([r['metrics'][c] for r in items]) for c in ('rmse','cvar')]))
                points.sort()
                if not points:continue
                array=np.array(points)
                for i,ax in enumerate(axes):
                    ax.plot(array[:,0],array[:,i+1],'.--' if delayed else '.-',
                            label=name+(' delayed' if delayed else ''),linewidth=1)
        for ax,metric in zip(axes,('RMSE (%)','CVaR95 loss (%)')):
            ax.set_ylabel(metric);ax.set_xlabel('Evaluation end (2025 = 2024-25)' if dimension=='year' else 'One-way cost (bp)')
        axes[-1].legend(fontsize=6,bbox_to_anchor=(1.02,1),loc='upper left')
        fig.tight_layout();fig.savefig(out/'evaluation'/filename,dpi=160,bbox_inches='tight');plt.close(fig)
    note=('本轮数据为已观察历史证据。指标分别按种子计算后比较；区间为共同到期月块重抽样，'
          '未作多重比较校正。0/10bp与一日延迟均为冻结5bp权重的闭环压力测试。'
          '三方向晋级只依据既定验证；未触发、数值失败和预算未完成必须分开报告。')
    note+='2023年同时用于主时期验证和一个滚动折的评价，该滚动折受方向筛选影响，不能作为独立确认。'
    table=['|方法|RMSE|CVaR95损失|平均交易成本|','|---|---:|---:|---:|']
    for key,items in groups.items():
        if key[:3]!=(2025,5,False):continue
        table.append('|'+key[3]+'|'+ '|'.join(f"{100*np.mean([r['metrics'][c] for r in items]):.4f}%"
                                                for c in ('rmse','cvar','cost'))+'|')
    text='# 瓶颈验证实验\n\n'+note+'\n\n'+'\n'.join(table)+'\n'
    diagnostic_summary={}
    cal=out/'markets/2025/calibration.json'
    if cal.exists():
        c=json.loads(cal.read_text(encoding='utf8'))
        diagnostic_summary['calibration']={k:c[k] for k in ('prediction_gate','stat','hedge','validation','gates')}
        if not c['prediction_gate']:
            text+='\n市场校准未通过预登记的预测与覆盖联合门槛，保留诊断负结果，不新增校准策略训练。\n'
    planner=out/'planner/2025/diagnostic.json'
    if planner.exists():
        p=json.loads(planner.read_text(encoding='utf8'))
        diagnostic_summary['planner']={k:p[k] for k in ('status','promote','by_seed','limitation') if k in p}
        if p.get('status')=='complete' and not p.get('promote'):
            text+='\n情景规划未通过模型内64状态晋级门槛；这不等价于完成了真实规划回测，也不证明所有规划方法无效。\n'
    write_json(out/'evaluation/diagnostic_summary.json',diagnostic_summary)
    risk_gate=out/'risk_validation_gate.json'
    if risk_gate.exists():
        g=json.loads(risk_gate.read_text(encoding='utf8'))
        write_json(out/'evaluation/risk_budget_selection.json',g)
        text+='\n风险预算晋级判定：'+('通过。' if g['promote'] else '未通过；保留全部可行及违约结果。')+'\n'
    (out/'evaluation/结果说明.md').write_text(text,encoding='utf8')
    (out/'evaluation/index.html').write_text('<!doctype html><meta charset="utf-8"><title>瓶颈验证实验</title>'
        '<style>body{max-width:1100px;margin:40px auto;font-family:system-ui}pre{white-space:pre-wrap}</style>'
        '<h1>瓶颈验证实验</h1><p>'+note+'</p><img src="risk_tradeoff.png" alt="RMSE CVaR tradeoff">'
        '<img src="rolling_stability.png" alt="Rolling evidence"><img src="cost_execution_stress.png" alt="Execution stress">'
        '<p><a href="metrics.csv">全部种子与成本结果</a> · <a href="subgroups.json">分组</a> · '
        '<a href="paired_intervals.json">配对区间</a></p><pre>'+text+'</pre>',encoding='utf8')
    target=project/'交付区/结果/瓶颈验证实验';target.mkdir(parents=True,exist_ok=True)
    shutil.copytree(out/'evaluation',target,dirs_exist_ok=True)
    code=project/'交付区/程序/瓶颈验证实验';code.mkdir(parents=True,exist_ok=True)
    for p in Path(__file__).parent.iterdir():
        if p.suffix in {'.py','.json','.md','.ps1'}:shutil.copyfile(p,code/p.name)
    parameters=project/'交付区/参数与依赖/瓶颈验证实验';parameters.mkdir(parents=True,exist_ok=True)
    for folder in ('policies','initial'):
        shutil.copytree(out/folder,parameters/folder,dirs_exist_ok=True)
    for name in ('registry.json','account_verification.json','risk_validation_gate.json',
                 'calibration_validation_gate.json','pipeline_status.json'):
        if (out/name).exists():shutil.copyfile(out/name,parameters/name)
    return dict(status='summary_generated',verified_replays=len(records),intervals=len(intervals),local=str(target))
