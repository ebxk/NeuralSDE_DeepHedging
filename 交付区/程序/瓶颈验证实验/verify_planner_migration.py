"""Verify the completed plain diagnostic is unchanged by fullstate/stress repairs.

No simulation or training. --apply narrowly updates only the two repaired source
hashes after deterministic old/new continuation comparisons and parent hash checks.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch


def verify(project,out,apply=False):
    sys.path.insert(0,str(project/'工作区/当前研究/程序/瓶颈验证实验'))
    import planner_experiment as new
    new.setup(project)
    from bottleneck_train import inputs_for
    from prepare_common import digest
    archive=out/'planner/review_fix_before_fullstate34'
    spec=importlib.util.spec_from_file_location('archived_plain_planner',archive/'planner_experiment.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    torch.set_num_threads(1)
    inputs=inputs_for(project,dict(end=2025,kind='plain'),out)
    manifest=json.loads((out/'planner/2025/diagnostic_states.json').read_text())
    checks=[];cached={}
    for seed in range(3):
        model=new.model_for(out,2025,seed)
        for number in (0,63):
            sample=manifest[number];year=inputs.folds[0 if sample['split']=='train' else 1]
            if year not in cached:
                paths,data=inputs.observed(year)
                cached[year]=(paths,new.augment_real(paths,data,inputs,out,2025))
            paths,data=cached[year]
            with torch.no_grad():
                a=old.state_snapshot(model,data,paths,inputs,sample['path_index'],sample['cut'],out,2025)
                b=new.state_snapshot(model,data,paths,inputs,sample['path_index'],sample['cut'],out,2025)
            if any(a[key]!=b[key] for key in a):raise AssertionError('Plain prefix account or market state changed')
            resultfile=out/'planner/2025/diagnostic'/f's{seed}_{number}.result.json'
            record=json.loads(resultfile.read_text())
            for count in (256,1024):
                arrays=dict(np.load(resultfile.with_suffix(f'.state.{count}.npz')))
                data_old=old.conditional_data(a,arrays)
                data_new=new.conditional_data(b,dict(arrays,delta=np.zeros_like(arrays['spot'],dtype='f'),
                    delta_valid=np.zeros_like(arrays['spot'],dtype=bool)))
                torch.testing.assert_close(data_old['f'],data_new['f'],rtol=0,atol=0)
                with torch.no_grad():
                    po=model.prepare(data_old);pn=model.prepare(data_new)
                    torch.testing.assert_close(po,pn,rtol=0,atol=0)
                    previous=torch.tensor([a['shares']/a['unit']])
                    wealth=torch.tensor([(a['cash']+a['receivable']+a['shares']*a['spot'])/a['notional']])
                    ao=model.action(po[0,:1],previous,wealth)[0]
                    an=model.action(pn[0,:1],previous,wealth)[0]
                    torch.testing.assert_close(ao,an,rtol=0,atol=0)
                    for action in (record['baseline_action'],record['selected_action']):
                        actions=torch.full((count,),action)
                        p0,c0=old.continuation_pnl(model,data_old,a,actions,.0005)
                        p1,c1=new.continuation_pnl(model,data_new,b,actions,.0005)
                        torch.testing.assert_close(p0,p1,rtol=0,atol=0)
                        torch.testing.assert_close(c0,c1,rtol=0,atol=0)
                checks.append(dict(seed=seed,state_number=number,scenarios=count,features_error=0,
                    prepared_error=0,action_error=0,pnl_error=0,cost_error=0,actions=2))
    stamp_path=out/'planner/2025/diagnostic.identity.json'
    original=json.loads((archive/'diagnostic.identity.json').read_text())
    current=json.loads(stamp_path.read_text())
    updated=json.loads(json.dumps(original))
    changed=[]
    for name,sha in original['files'].items():
        actual=digest(Path(name))
        if actual!=sha:
            if Path(name).name not in ('planner_experiment.py','fullstate_policy.py'):
                raise AssertionError(f'An unreviewed parent changed: {name}')
            changed.append(dict(path=name,before=sha,after=actual));updated['files'][name]=actual
    previous_report_path=out/'planner/2025/diagnostic_migration.json'
    if current not in (original,updated):
        previous=json.loads(previous_report_path.read_text())
        known=json.loads(json.dumps(original))
        if not previous.get('applied') or previous['old_source_sha256']!=digest(archive/'planner_experiment.py'):
            raise AssertionError('Unverified prior migration')
        for change in previous['changed_sources']:
            if Path(change['path']).name not in ('planner_experiment.py','fullstate_policy.py'):
                raise AssertionError('Prior migration changed an unreviewed parent')
            known['files'][change['path']]=change['after']
        if current!=known:raise AssertionError('Unknown diagnostic identity version')
        new.write(archive/f'migration_{digest(previous_report_path)[:12]}.json',previous)
        new.write(archive/f'migration_{digest(previous_report_path)[:12]}.identity.json',current)
    report=dict(status='complete',checks=checks,changed_sources=changed,applied=apply,
        method='Old/new exact comparisons on six states spanning all seeds and train/validation, both scenario banks, two actions.',
        reason='Plain DensePolicy reads only base8+actual position. Added fullstate inputs and execution stresses do not change its default 5bp continuation.',
        old_source=str(archive/'planner_experiment.py'),old_source_sha256=digest(archive/'planner_experiment.py'),
        archived_result_sha256=digest(archive/'diagnostic.json'),
        restriction='Migration applies only to the already completed plain diagnostic, never to new fullstate training or scenario caches.')
    new.write(out/'planner/2025/diagnostic_migration.json',report)
    if apply:
        new.write(stamp_path,updated)
        result=json.loads((archive/'diagnostic.json').read_text())
        result['identity_migration']='diagnostic_migration.json'
        new.write(out/'planner/2025/diagnostic.json',result)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',required=True,type=Path);parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args();print(json.dumps(verify(args.project.resolve(),args.out.resolve(),args.apply),indent=2))
