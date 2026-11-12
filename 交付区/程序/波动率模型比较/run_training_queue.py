"""Generation-aware recovery; a rebuilt generator supersedes blocked jobs."""
import argparse
import json
import time
import traceback
from datetime import datetime,timezone
from study_core import OUT,HERE,CONFIG,registry,write_json
from artifact_guard import verify_ready,stable_digest,check_hashes
from train_common import train_one

def archive_job(job):
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    dest=OUT/'attempts'/f"policy_{job['key']}_{stamp}";dest.mkdir(parents=True,exist_ok=True)
    for folder,suffix in [('policies','.json'),('policies','.pt'),('policies','.progress.json'),('failures','.json')]:
        p=OUT/folder/f"{job['key']}{suffix}"
        if p.exists():
            if not p.resolve().is_relative_to(OUT.resolve()) or not dest.resolve().is_relative_to(OUT.resolve()):raise ValueError('Archive outside study')
            p.rename(dest/p.name)

def context(job):
    ready=OUT/f"inputs/{job['end']}/sim_{job['model']}.ready.json"
    return dict(ready=stable_digest(ready),trainer=stable_digest(HERE/'train_common.py'),
        policy=stable_digest(HERE/'common_policy.py'))

def resolved(job):
    receipt=OUT/'policies'/f"{job['key']}.json";failure=OUT/'failures'/f"{job['key']}.json"
    ready=OUT/f"inputs/{job['end']}/sim_{job['model']}.ready.json"
    artifact=ready.with_name(f"sim_{job['model']}.npz")
    valid=False
    if ready.exists():
        try:verify_ready(ready,artifact,CONFIG,job['model'],job['end']);valid=True
        except (ValueError,OSError):pass
    if receipt.exists() and valid:
        try:
            r=json.loads(receipt.read_text(encoding='utf8'))
            check_hashes(r['identity']['parents'])
            if r['identity']['job']!=job or r['identity']['config']!=json.loads(json.dumps(CONFIG)):raise ValueError('Changed policy identity')
            if stable_digest(OUT/f"initial/{job['initial_key']}.pt")!=r['identity']['initial_sha256']:raise ValueError('Changed matched initial weights')
            if stable_digest(receipt.with_suffix('.pt'))!=r['weight_sha256']:raise ValueError('Changed weights')
            return 'done'
        except (ValueError,OSError,KeyError):return 'train'
    if valid:
        if failure.exists():
            f=json.loads(failure.read_text(encoding='utf8'))
            if f.get('status')=='training_failed_current_generation' and f.get('context')==context(job):return 'failed'
        return 'train'
    if failure.exists() and json.loads(failure.read_text(encoding='utf8')).get('status')=='blocked_by_generator':return 'blocked'
    return 'wait'

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);args=p.parse_args()
    tasks=[j for j in registry() if j['seed']==args.seed]
    while True:
        state={j['key']:resolved(j) for j in tasks}
        for job in tasks:
            if state[job['key']]!='train':continue
            if (OUT/'policies'/f"{job['key']}.json").exists() or (OUT/'failures'/f"{job['key']}.json").exists():archive_job(job)
            print('START',job['key'],flush=True)
            try:train_one(job);state[job['key']]='done'
            except Exception:
                error=__import__('sys').exc_info()[1]
                result=dict(key=job['key'],status='training_failed_current_generation',context=context(job),
                    category='numerical_rejection' if isinstance(error,FloatingPointError) else 'infrastructure_failure',reason=traceback.format_exc())
                write_json(OUT/'failures'/f"{job['key']}.json",result);state[job['key']]='failed'
            print('FINISH',job['key'],state[job['key']],flush=True)
            write_json(OUT/f'queue_seed{args.seed}.json',dict(complete=False,states=state))
        sealed=(OUT/'generators_finished.json').exists()
        complete=sealed and all(v in ('done','failed','blocked') for v in state.values())
        write_json(OUT/f'queue_seed{args.seed}.json',dict(complete=complete,states=state))
        if complete:return
        time.sleep(30)

if __name__=='__main__':
    try:main()
    except BaseException:
        seed=__import__('sys').argv[-1]
        write_json(OUT/f'queue_seed{seed}.producer_failure.json',dict(complete=False,reason=traceback.format_exc()))
        raise
