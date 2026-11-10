"""Verify every completed unconditional checkpoint after a narrow identity repair."""
import bootstrap
import json
import shutil
from pathlib import Path
from bottleneck_core import jobs
from run_pipeline import train_stage
from prepare_common import digest,write_json
from finalize_run import finalize

def main():
    project=bootstrap.PROJECT;out=project/'工作区/数据/处理后/瓶颈验证实验'
    cfg=json.loads((Path(__file__).parent/'study.json').read_text(encoding='utf8'))
    # No training is permitted here: any missing artifact must fail before dispatch.
    selected=[j for j in jobs() if j['trigger']=='always']
    for task in selected:
        p=out/'policies'/f"{task['key']}.pt"
        s=json.loads(p.with_suffix('.json').read_text(encoding='utf8'))
        assert s['status']=='complete' and s['weights_sha256']==digest(p)
    ok=train_stage(project,out,cfg,'foundational_revalidated',selected)
    result=json.loads((out/'foundational_revalidated.json').read_text(encoding='utf8'))
    assert ok and all(r['status']=='cached' for r in result['results'])
    archive=out/'acceptance_history';archive.mkdir(exist_ok=True)
    for name in ('pipeline_status.json','acceptance.json'):
        p=out/name
        if p.exists():shutil.copyfile(p,archive/f'{p.stem}.{digest(p)[:12]}.json')
    status=json.loads((out/'pipeline_status.json').read_text(encoding='utf8'))
    status['stages']['foundational']=True
    status['foundational_revalidation']='foundational_revalidated.json'
    status['identity_repair']='Only equivalent main episode path metadata; weights and evaluation unchanged'
    write_json(out/'pipeline_status.json',status)
    finalize(project,out)

if __name__=='__main__':main()
