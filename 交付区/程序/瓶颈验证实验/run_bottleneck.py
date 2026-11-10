"""Resume the frozen study; status never equates a launched job with completion."""
import bootstrap
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback
from bottleneck_core import jobs
from bottleneck_train import train_one
from prepare_common import write_json

def group(project,out,cfg,seed,tasks):
    errors=[]
    for task in tasks:
        try:
            print('START',task['key'],flush=True)
            result=train_one(project,out,cfg,task)
            print('DONE',task['key'],result,flush=True)
        except Exception:
            error=dict(task=task,reason=traceback.format_exc())
            write_json(out/'failures'/f"{task['key']}.json",error)
            errors.append(error)
            print('FAILED',task['key'],error['reason'],flush=True)
    return errors

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project',type=Path,default=bootstrap.PROJECT)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--phase',choices=['foundational','risk_main'],default='foundational')
    args=parser.parse_args()
    project=args.project.resolve()
    out=project/'工作区/数据/处理后/瓶颈验证实验'
    for folder in ('policies','initial','failures','evaluation'):
        (out/folder).mkdir(parents=True,exist_ok=True)
    cfg=json.loads((Path(__file__).parent/'study.json').read_text(encoding='utf8'))
    registry=jobs()
    manifest=dict(config=cfg,jobs=registry,maximum=117)
    target=out/'registry.json'
    if target.exists() and json.loads(target.read_text(encoding='utf8'))!=manifest:
        raise ValueError('Frozen manifest changed; use a new study identity')
    if not target.exists():
        write_json(target,manifest)
    selected=[j for j in registry if j['trigger']==('always' if args.phase=='foundational' else 'risk_main')]
    # Within a seed: ordinary matching baselines precede targets depending on them.
    errors=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(group,project,out,cfg,s,[j for j in selected if j['seed']==s]) for s in range(3)]
        for future in as_completed(futures):
            errors.extend(future.result())
    complete=sum((out/'policies'/f"{j['key']}.json").exists() for j in selected)
    write_json(out/f'{args.phase}.json',dict(expected=len(selected),complete=complete,
        failures=errors,finished=datetime.now(timezone.utc).isoformat()))
    if errors:
        raise RuntimeError(f'{len(errors)} tasks failed; see failures directory')

if __name__=='__main__':
    main()
