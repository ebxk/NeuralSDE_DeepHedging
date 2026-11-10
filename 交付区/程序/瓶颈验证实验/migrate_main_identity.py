"""One narrow, audited migration: corrected episode location, identical main bytes."""
import bootstrap
import json
import shutil
import torch
from pathlib import Path
from bottleneck_train import inputs_for,identity,job_for,make_model_for,evaluate
from prepare_common import digest,write_json

OLD='fc7892951883eda5da83b7529b0764b324b29e42d0c60fe41360b79d2782e57d'

def migrate(project,out,cfg,task):
    if task['end']!=2025 or task['kind'] not in {'plain','teacher'}:return
    weights=out/'policies'/f"{task['key']}.pt";status=weights.with_suffix('.json')
    if not status.exists():return
    previous=json.loads(status.read_text(encoding='utf8'))
    key=str(Path('瓶颈验证实验')/'bottleneck_train.py')
    old=project/'工作区/数据/处理后/完整认购/episode_days.csv'
    new=project/'工作区/数据/处理后/创新组合实验/2025/episode_days.csv'
    current=Path(__file__).parent/'bottleneck_train.py'
    recorded_source=previous['identity']['source'].get(key)
    if recorded_source not in (OLD,digest(current)):return
    if recorded_source==digest(current) and str(old) not in previous['identity']['inputs']:return
    assert digest(old)==digest(new)
    snapshot=out/'foundational_train_v1.py'
    assert digest(snapshot)==OLD
    before=snapshot.read_text(encoding='utf8');after=current.read_text(encoding='utf8')
    assert before.replace("Inputs(project/'工作区/数据/处理后/完整认购',folder,rep,",
                          "Inputs(old/str(end),folder,rep,")==after
    inputs=inputs_for(project,task,out)
    initial=out/'initial'/f"2025_wide_s{task['seed']}.pt"
    expected=identity(project,task,cfg,inputs,initial)
    transformed=json.loads(json.dumps(previous['identity']))
    transformed['source'][key]=expected['source'][key]
    transformed['inputs'][str(new)]=transformed['inputs'].pop(str(old))
    assert transformed==expected and digest(weights)==previous['weights_sha256']
    model=make_model_for(job_for(task));model.load_state_dict(torch.load(weights,weights_only=True))
    _,validation=inputs.observed(2023)
    actual=evaluate(model,job_for(task),validation)
    for name in actual:assert abs(actual[name]-previous['validation'][name])<1e-10
    archive=out/'identity_migration';archive.mkdir(exist_ok=True)
    saved=archive/status.name
    if saved.exists() and digest(saved)!=digest(status):
        saved=archive/f'{status.stem}.{digest(status)[:12]}.json'
    shutil.copyfile(status,saved)
    previous['identity']=expected
    previous['migration']=dict(reason='Episode path correction; main file bytes identical',
                                old_source=recorded_source,new_source=expected['source'][key],
                                equivalence_snapshot=OLD,validation_checked=True)
    write_json(status,previous)
