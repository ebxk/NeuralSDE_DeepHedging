"""Dependency-ordered, checkpointed execution with validation-only promotion."""
import bootstrap
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import ctypes
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
from bottleneck_core import jobs
from bottleneck_train import train_one
from migrate_main_identity import migrate
from prepare_common import write_json,digest

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))

def wait_pid(pid):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenProcess(0x100000,False,pid)
    if handle:
        try:kernel.WaitForSingleObject(handle,0xffffffff)
        finally:kernel.CloseHandle(handle)

def train_group(project,out,cfg,tasks):
    result=[]
    for task in tasks:
        try:
            print('TRAIN START',task['key'],flush=True)
            migrate(project,out,cfg,task)
            result.append(dict(task=task,status=train_one(project,out,cfg,task)))
            print('TRAIN COMPLETE',task['key'],flush=True)
        except Exception:
            failure=dict(task=task,status='failed',reason=traceback.format_exc())
            write_json(out/'failures'/f"{task['key']}.retry.json",failure)
            result.append(failure)
            print('TRAIN FAILED',task['key'],failure['reason'],flush=True)
    return result

def train_stage(project,out,cfg,name,tasks):
    from source_guard import freeze
    freeze(project,out)
    results=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        fs=[pool.submit(train_group,project,out,cfg,[j for j in tasks if j['seed']==s]) for s in range(3)]
        for f in as_completed(fs):results.extend(f.result())
    write_json(out/f'{name}.json',dict(expected=len(tasks),results=results,
        complete=sum(x['status']!='failed' for x in results),failures=sum(x['status']=='failed' for x in results)))
    return all(x['status']!='failed' for x in results)

def calibration_gate(out):
    reference=[read(out/'policies'/f'2025_plain_s{s}.json')['validation'] for s in range(3)]
    results=[]
    for kind in ('stat','hedge'):
        candidate=[read(out/'policies'/f'2025_{kind}_s{s}.json')['validation'] for s in range(3)]
        effects={k:float((np.mean([r[k] for r in candidate])-np.mean([r[k] for r in reference]))/max(abs(np.mean([r[k] for r in reference])),1e-12))
                 for k in ('rmse','cvar','cost')}
        directions={k:sum(c[k]<r[k] for c,r in zip(candidate,reference)) for k in ('rmse','cvar')}
        eligible=any(effects[k]<=-.01 and directions[k]>=2 and effects[other]<=.05 and effects['cost']<=.05
                     for k,other in (('rmse','cvar'),('cvar','rmse')))
        results.append(dict(kind=kind,effects=effects,directions=directions,promote=eligible))
    return dict(promote=any(r['promote'] for r in results),methods=results)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait-pid',type=int,default=0)
    args=parser.parse_args();project=bootstrap.PROJECT
    out=project/'工作区/数据/处理后/瓶颈验证实验';here=Path(__file__).parent
    cfg=read(here/'study.json');registry=jobs();stages={};errors=[]
    def status(phase,**extra):
        write_json(out/'pipeline_status.json',dict(phase=phase,stages=stages,errors=errors,
            updated=datetime.now(timezone.utc).isoformat(),complete=False,**extra))
        print('PHASE',phase,flush=True)
    def market(end):
        command=[str(project/'工作区/运行支持/环境/market/python.exe'),'-u',str(here/'market_bridge.py'),
                 '--project',str(project),'--out',str(out),'--end',str(end),'--export']
        environment=dict(os.environ)
        environment.pop('PYTHONHOME',None)
        environment.pop('PYTHONPATH',None)
        subprocess.run(command,check=True,env=environment)
        return read(out/'markets'/str(end)/'calibration.json')
    def planner(end,stage):
        command=[sys.executable,'-u',str(here/'planner_experiment.py'),
                 '--project',str(project),'--out',str(out),'--end',str(end),'--stage',stage]
        subprocess.run(command,check=True)
        result=read(out/'planner'/str(end)/f'{stage}.json')
        if result.get('status')!='complete':
            stages[f'planner_{end}_{stage}']=result
            raise RuntimeError(f"Planner stage incomplete: {result.get('status')}")
        return result
    if args.wait_pid:
        status('waiting_for_initial_foundation');wait_pid(args.wait_pid)
    status('foundational_recovery')
    stages['foundational']=train_stage(project,out,cfg,'foundational_verified',[j for j in registry if j['trigger']=='always'])
    # Independent branches keep their failures, without deleting unrelated jobs.
    status('risk_main')
    stages['risk_main']=train_stage(project,out,cfg,'risk_main',[j for j in registry if j['trigger']=='risk_main'])
    if stages['risk_main']:
        try:
            from validation_controls import static_controls,budget_gate
            records=[]
            for seed in range(3):
                r=static_controls(project,out,2025,seed)
                for kind in ('lambda05','lambda10','budget02','budget05'):
                    r[kind]=read(out/'policies'/f'2025_{kind}_s{seed}.json')['validation']
                records.append(r)
            gate=budget_gate(records)
            write_json(out/'risk_validation_gate.json',dict(**gate,records=records))
            stages['risk_promoted']=gate['promote']
            if gate['promote']:
                status('risk_rolling')
                stages['risk_rolling']=train_stage(project,out,cfg,'risk_rolling',[j for j in registry if j['trigger']=='risk_rolling'])
        except Exception:errors.append(dict(branch='risk_validation',reason=traceback.format_exc()))
    try:
        status('market_prediction')
        runner=out/'calibration_runner.json'
        if runner.exists():wait_pid(read(runner)['pid'])
        cal=market(2025)
        stages['prediction_promoted']=cal['prediction_gate']
        if cal['prediction_gate']:
            stages['calibration_main']=train_stage(project,out,cfg,'calibration_main',[j for j in registry if j['trigger']=='prediction'])
            if stages['calibration_main']:
                gate=calibration_gate(out);write_json(out/'calibration_validation_gate.json',gate)
                stages['calibration_promoted']=gate['promote']
                if gate['promote']:
                    for end in (2021,2022,2023):
                        market(end)
                        # Promotion was fixed on the main validation: retain bad rolling years.
                        stages[f'calibration_rolling_{end}']=train_stage(project,out,cfg,f'calibration_rolling_{end}',
                            [j for j in registry if j['trigger']=='calibration_rolling' and j['end']==end])
    except Exception:errors.append(dict(branch='calibration',reason=traceback.format_exc()))
    try:
        status('planner_diagnostic')
        diagnostic=planner(2025,'diagnostic');stages['planner_diagnostic']=diagnostic
        if diagnostic.get('promote'):
            planner(2025,'train');validation=planner(2025,'validate')
            stages['planner_validation']=validation
            if validation.get('promote'):
                for end in (2025,2021,2022,2023):
                    if end!=2025:planner(end,'train');planner(end,'validate')
                    stages[f'planner_evaluation_{end}']=planner(end,'evaluate')
    except Exception:errors.append(dict(branch='planner',reason=traceback.format_exc()))
    # All promotion decisions are fixed before reading historical evaluation PnL.
    selected=[]
    for task in registry:
        p=out/'policies'/f"{task['key']}.json"
        if p.exists() and task['kind']!='fullstate' and read(p).get('status')=='complete':selected.append(task)
    write_json(out/'evaluation/frozen_selection.json',dict(stages=stages,
        policies={j['key']:read(out/'policies'/f"{j['key']}.json")['weights_sha256'] for j in selected}))
    status('historical_evaluation')
    from bottleneck_report import replay_task
    rows=[]
    for task in selected:
        try:rows.extend(replay_task(project,out,task))
        except Exception:errors.append(dict(branch='historical',task=task,reason=traceback.format_exc()))
    write_json(out/'evaluation/metrics.json',rows)
    from finish_report import finish
    stages['report']=finish(project,out)
    write_json(out/'pipeline_status.json',dict(phase='finished',stages=stages,errors=errors,
        complete=not errors and all(v is not False for k,v in stages.items() if 'promoted' not in k),
        finished=datetime.now(timezone.utc).isoformat(),trained=len(selected),maximum_registered=117,
        evaluation_files=len(rows),
        acceptance='Verified ledger replays, common month blocks, benchmarks and local report generated',
        note='Untriggered branches are not failed training jobs.'))

if __name__=='__main__':main()
