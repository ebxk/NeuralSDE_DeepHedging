"""Independent final acceptance after the runner exits; no training is performed."""
import bootstrap
import argparse
import json
from pathlib import Path
from prepare_common import digest,write_json
from run_pipeline import wait_pid
from source_guard import freeze

def finalize(project,out):
    status=json.loads((out/'pipeline_status.json').read_text(encoding='utf8'))
    errors=list(status.get('errors',[]));stages=status.get('stages',{})
    if status.get('phase')!='finished':errors.append(dict(branch='runner',reason='Runner did not reach final evaluation'))
    for key,value in stages.items():
        if value is False and 'promoted' not in key:errors.append(dict(branch=key,reason='Stage incomplete'))
        if isinstance(value,dict) and value.get('status') in {'failed','budget_incomplete'}:
            errors.append(dict(branch=key,reason=value['status']))
    freeze(project,out)
    expected=45 # 33 unconditional and 12 main risk controls
    if stages.get('prediction_promoted'):expected+=6
    if stages.get('calibration_promoted'):expected+=18
    if stages.get('risk_promoted'):expected+=36
    diagnostic=stages.get('planner_diagnostic',{})
    validation=stages.get('planner_validation',{})
    if diagnostic.get('promote'):expected+=3
    if validation.get('promote'):expected+=9
    registry=json.loads((out/'registry.json').read_text(encoding='utf8'))['jobs']
    finished=[]
    for task in registry:
        weights=out/'policies'/f"{task['key']}.pt";marker=weights.with_suffix('.json')
        if marker.exists() and weights.exists():
            data=json.loads(marker.read_text(encoding='utf8'))
            if data.get('status')=='complete' and data.get('weights_sha256')==digest(weights):finished.append(task['key'])
    if len(finished)!=expected:errors.append(dict(branch='manifest',reason=f'Expected {expected} triggered weights, found {len(finished)}'))
    needed=['metrics.csv','paired_intervals.json','subgroups.json','risk_tradeoff.png','结果说明.md','index.html',
            'diagnostic_summary.json','risk_budget_selection.json']
    for name in needed:
        if not (out/'evaluation'/name).exists():errors.append(dict(branch='delivery',reason=f'Missing {name}'))
    if validation.get('promote'):
        catalog={}
        for p in (out/'evaluation/paths').glob('*.json'):
            item=json.loads(p.read_text(encoding='utf8'));m=item.get('metrics',{})
            if m:catalog[m['end'],m['kind'],m['seed'],m['cost_bp'],m['delay']]=item
        for end in (2025,2021,2022,2023):
            for seed in range(3):
                reference=catalog.get((end,'plain',seed,5,False))
                reference_ids=[r['episode'] for r in reference['rows']] if reference else None
                for kind in ('planner','fullstate'):
                    for cost in (0,5,10):
                        for delay in (False,True):
                            key=(end,kind,seed,cost,delay);item=catalog.get(key)
                            if item is None or [r['episode'] for r in item['rows']]!=reference_ids:
                                errors.append(dict(branch='planner_delivery',reason=f'Missing/common-path mismatch: {key}'))
    verification=json.loads((out/'account_verification.json').read_text(encoding='utf8'))
    if verification.get('paths')!=773 or max(verification[k] for k in
        ('original_parity','continuation_error','independent_ledger_error'))>1e-8:
        errors.append(dict(branch='account',reason='Real account acceptance failed'))
    receipt=dict(complete=not errors,triggered=expected,completed_weights=len(finished),
                 maximum_registered=117,errors=errors,stages=stages)
    write_json(out/'acceptance.json',receipt)
    status['complete']=not errors;status['acceptance']=receipt
    write_json(out/'pipeline_status.json',status)
    target=project/'交付区/结果/瓶颈验证实验'
    if target.exists():write_json(target/'completion.json',receipt)
    print(json.dumps(receipt,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait-pid',type=int,default=0);a=p.parse_args()
    if a.wait_pid:wait_pid(a.wait_pid)
    finalize(bootstrap.PROJECT,bootstrap.PROJECT/'工作区/数据/处理后/瓶颈验证实验')
