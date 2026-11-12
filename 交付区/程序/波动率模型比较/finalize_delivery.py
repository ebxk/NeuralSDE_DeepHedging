"""Validate saved selections and deliver aggregates, never raw quotes to Git."""
import json
import shutil
import subprocess
import numpy as np
import pandas as pd
import torch
from study_core import HERE,OUT,PROJECT,CONFIG,registry,write_json,digest,bootstrap_legacy
from common_policy import CommonPolicy
from train_common import load_data,subset
from artifact_guard import require_delivery_manifest,check_hashes,verify_ready,_hashed_version

AGGREGATE_RESULTS=('seed_summary.csv','overall.csv','group_metrics.csv','paired_intervals.csv',
    'historical_references.csv','surface_comparison.csv','surface_status.json','简短结论.md','report_status.json',
    'direct_summary.csv','prediction_summary.csv','numerical_summary.csv',
    'numerical_diagnostics.csv','compute_costs.csv','numerical_status.json','comparison_figures.json')

def copy_local_parameters(params):
    """Copy fitted parameters; reference large simulation/diagnostic arrays in place."""
    references=[]
    for source,name in [('policies','weights'),('markets','markets'),('representations','representations'),
                        ('initial','initial'),('inputs','inputs'),('predictions','prediction_diagnostics'),
                        ('factor_quote_audit','factor_audits'),('evaluation','evaluation')]:
        if not (OUT/source).exists():continue
        for path in sorted((OUT/source).rglob('*')):
            if not path.is_file():continue
            copy=(source in ('policies','representations','initial') or
                  source=='markets' and (path.suffix in ('.json','.h5','.pt') or path.name=='states.csv'))
            if copy:
                target=params/name/path.relative_to(OUT/source);target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(path,target)
            else:references.append(dict(path=str(path.resolve()),sha256=digest(path),bytes=path.stat().st_size))
    write_json(params/'local_artifact_references.json',dict(artifacts=references))
    for name in ('acceptance_manifest.json','registry.json','account_acceptance.json'):
        shutil.copy2(OUT/name,params/name)
    for name,target_name in [('shared_surfaces.npz','shared_surfaces.npz'),('clean_base/status.json','status.json')]:
        target=params/'clean_base'/target_name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(OUT/'data'/name,target)
    for name in ('requirements-lock.txt','requirements-market-lock.txt'):shutil.copy2(HERE/name,params/name)

def collect_aggregate_diagnostics():
    """Only summarize already saved diagnostics, with no scientific refitting."""
    bootstrap_legacy();from train import expected_shortfall
    direct=[];prediction=[];numerical=[]
    for _,_,end in CONFIG['folds']:
        for folder,file,model in [(OUT/f'direct/{end}','group_metrics.csv','static'),
                *[(OUT/f'direct_dynamic/{end}/{m}','groups.csv',m) for m in ('heston','sabr','localvol')]]:
            if (folder/file).exists():
                frame=pd.read_csv(folder/file).rename(columns={'bp':'cost_bp','hedge':'method','delta':'method',
                    'mean_cost':'cost','mean_turnover':'turnover'});frame['end']=end
                if 'model' not in frame:frame['model']=model
                direct.extend(frame.to_dict('records'))
        file=OUT/f'direct_nsde/{end}/pnl.csv'
        if file.exists():
            frame=pd.read_csv(file)
            for (method,bp,execution),g in frame.groupby(['method','cost_bp','execution']):
                direct.append(dict(end=end,model='nsde',method=method,cost_bp=bp,execution=execution,paths=len(g),
                    mean=float(g.pnl.mean()),std=float(g.pnl.std(ddof=0)),
                    cvar=float(expected_shortfall(-g.pnl.to_numpy())),
                    rmse=float(np.sqrt(np.mean(g.pnl**2))),cost=float(g.cost.mean()),
                    turnover=float(g.turnover.mean()) if 'turnover' in g else None,
                    decision_clips=int(g.clip_decisions.sum())))
        for model in CONFIG['models']:
            file=OUT/f'predictions/{end}_{model}.csv'
            if file.exists():
                frame=pd.read_csv(file)
                for horizon,g in frame.groupby('horizon'):
                    row=dict(end=end,model=model,horizon=int(horizon),origins=len(g),scenarios=int(g.n.sum()))
                    for metric in ('residual_crps','covered95','energy_score_raw_units'):
                        if metric in g:row['mean_'+metric]=float(g[metric].mean())
                    prediction.append(row)
            file=OUT/f'markets/{end}/{model}/market.json'
            if file.exists():
                r=json.loads(file.read_text(encoding='utf8'));numeric=r.get('numeric',{})
                numerical.append(dict(end=end,model=model,stage='joint_market',complete=r.get('complete',r.get('status')=='complete'),
                    validation_nll=r.get('validation_nll'),full_slack=numeric.get('full_slack'),fine_slack=numeric.get('fine_slack')))
            file=OUT/f'markets/{end}/{model}/global.json'
            if file.exists():
                r=json.loads(file.read_text(encoding='utf8'))['model']['diagnostics']
                numerical.append(dict(end=end,model=model,stage='global_Q',complete=r['valid'],objective=r['objective'],
                    qualified_candidates=r.get('qualified_candidates'),optimizer_success=r.get('optimizer_success')))
    for name,rows in [('direct_summary.csv',direct),('prediction_summary.csv',prediction),('numerical_summary.csv',numerical)]:
        if rows:pd.DataFrame(rows).to_csv(OUT/'evaluation'/name,index=False)

def run():
    _hashed_version.cache_clear()
    acceptance=require_delivery_manifest(OUT/'acceptance_manifest.json')
    bootstrap_legacy();from extended_policy import rollout
    from train import score
    torch.set_num_threads(2)
    pending=[];failed=[];verified=[]
    for job in registry():
        receipt=OUT/'policies'/f"{job['key']}.json"
        if not receipt.exists():
            (failed if (OUT/'failures'/f"{job['key']}.json").exists() else pending).append(job['key']);continue
        record=json.loads(receipt.read_text(encoding='utf8'));weight=receipt.with_suffix('.pt')
        check_hashes(record['identity']['parents'])
        initial=OUT/'initial'/f"{job['initial_key']}.pt"
        if digest(initial)!=record['identity']['initial_sha256']:raise ValueError('Matched initialization changed')
        if digest(weight)!=record['weight_sha256']:raise ValueError('Model weight changed')
        root=OUT/f"inputs/{job['end']}";scale=dict(np.load(root/'feature_scale.npz'))
        verify_ready(root/f"sim_{job['model']}.ready.json",root/f"sim_{job['model']}.npz",CONFIG,job['model'],job['end'])
        data=load_data(root/f"real_{job['fold'][1]}.npz",**scale)
        model=CommonPolicy(job['information']=='surface');model.load_state_dict(torch.load(weight,weights_only=True))
        with torch.no_grad():
            p=torch.cat([rollout(model,subset(data,slice(i,i+256)),.0005)[0]
                for i in range(0,len(data['f']),256)]).numpy()
        error=abs(score(p,job['lam'])-record['validation'])
        if error>1e-8:raise ValueError(f"Reload validation mismatch {job['key']}: {error}")
        verified.append(dict(key=job['key'],validation_error=error,weight_sha256=digest(weight)))
    if pending:raise ValueError(f'{len(pending)} unexecuted training tasks')
    account=json.loads((OUT/'account_acceptance.json').read_text(encoding='utf8'))
    if not account['complete']:raise ValueError('Account acceptance incomplete')
    subprocess.run([str(PROJECT/'工作区/运行支持/环境/volatility/Scripts/python.exe'),'-m','unittest',
        'discover','-s',str(HERE),'-p','test_*.py'],check=True)
    table=pd.read_csv(OUT/'evaluation/seed_summary.csv')
    main=table[(table.end==2025)&(table.cost_bp==5)&table.execution.eq('same_close')]
    lines=['# 本轮简短结果说明','',
        f'登记432项，已训练并重载核验{len(verified)}项，失败/生成器阻断{len(failed)}项；全部原实验保留。',
        '','主时期为已观察的2024—2025历史行情。以下为5bp、同收盘成交的实际完成种子均值，按初始名义金额归一化；种子数见完整表。',
        '','|市场|信息|目标λ|RMSE|CVaR95损失|平均成本|','|---|---|---:|---:|---:|---:|']
    lines[4:4]=(HERE/'结果解读.md').read_text(encoding='utf8').splitlines()+['']
    for _,r in main.iterrows():
        lines.append(f'|{r.model}|{r.information}|{r.lam:g}|{100*r.rmse_mean:.4f}%|{100*r.cvar_mean:.4f}%|{100*r.cost_mean:.4f}%|')
    lines.extend(['','不能由曲面拟合更准直接推断对冲更好。请同时读取曲面留报价、匹配NSDE配对区间、跨年分组与成本/执行压力表。',
        '置信区间跨零表示证据有限；全部比较是探索性的，未作多重比较调整，主时期只有24个到期月。',
        'SVI系列结果属于曲面加历史联合动态；局部波动率属于指定历史漂移代理；有限格点检查不构成动态无套利证明。',
        '条件预测使用训练起点及保留模拟情景，属于训练内部诊断，不能当作全新时间外预测证明。',
        '','本地逐日账本和逐路径结果：工作区/数据/处理后/波动率模型比较/evaluation。'])
    if failed:lines.extend(['','未完成/失败条目：',*['- '+k for k in failed]])
    (OUT/'evaluation/简短结论.md').write_text('\n'.join(lines),encoding='utf8')
    delivery=PROJECT/'交付区';program=delivery/'程序/波动率模型比较';result=delivery/'结果/波动率模型比较'
    params=delivery/'参数与依赖/波动率模型比较'
    for folder in (program,result,params):folder.mkdir(parents=True,exist_ok=True)
    collect_aggregate_diagnostics()
    for p in HERE.iterdir():
        if p.suffix in ('.py','.md','.txt'):shutil.copy2(p,program/p.name)
    for p in (OUT/'evaluation').iterdir():
        if p.name in AGGREGATE_RESULTS or p.suffix=='.png':
            shutil.copy2(p,result/p.name)
    write_json(params/'configuration.json',CONFIG)
    shutil.copy2(OUT/'registry.json',params/'registry.json')
    copy_local_parameters(params)
    write_json(result/'completion.json',dict(execution_complete=True,all_models_succeeded=not failed,
        registered=432,verified_weights=len(verified),failed=failed,pending=[],account=account,
        limitations=['observed historical evaluation','training-internal forecast diagnostics'],
        detailed_local_artifacts='工作区/数据/处理后/波动率模型比较'))
    write_json(result/'acceptance_summary.json',dict(complete=acceptance['complete'],
        coverage_count=len(acceptance['coverage']),declared_failures=[{'stage':r['stage'],'reason':r['reason']} for r in acceptance['declared_failures']],
        manifest_sha256=digest(OUT/'acceptance_manifest.json')))
    write_json(OUT/'reload_acceptance.json',dict(complete=True,weights=verified,failed=failed))

if __name__=='__main__':run()
