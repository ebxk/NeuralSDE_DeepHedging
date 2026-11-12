"""Fail-closed artifact provenance and completed-study coverage checks."""
import json
from functools import lru_cache
from pathlib import Path
from study_core import HERE,digest

@lru_cache(maxsize=4096)
def _hashed_version(path,mtime,size):
    return digest(path)

def stable_digest(path):
    p=Path(path);s=p.stat()
    return _hashed_version(str(p.resolve()),s.st_mtime_ns,s.st_size)

def check_hashes(mapping):
    for name,expected in mapping.items():
        p=Path(name)
        if not p.is_absolute():p=HERE/p
        if not p.is_file() or stable_digest(p)!=expected:raise ValueError(f'Stale artifact parent/source: {p}')

def verify_ready(ready,artifact,expected_config=None,model=None,end=None):
    r=json.loads(Path(ready).read_text(encoding='utf8'))
    if r.get('complete') is not True:raise ValueError('Incomplete generator readiness')
    if not Path(artifact).is_file() or stable_digest(artifact)!=r.get('input_sha256'):
        raise ValueError('Simulator bytes differ from readiness receipt')
    check_hashes(r.get('parents',{}))
    sources=r.get('sources') or r.get('identity',{}).get('sources')
    if not sources:raise ValueError('Missing producer source provenance')
    check_hashes(sources)
    config=r.get('config',r.get('identity',{}).get('config'))
    if expected_config is not None and config!=json.loads(json.dumps(expected_config)):
        raise ValueError('Simulator configuration identity changed')
    if model is not None and r.get('model')!=model:raise ValueError('Wrong generator model')
    if end is not None and r.get('end')!=end:raise ValueError('Wrong generator fold')
    side=r.get('state_sidecar')
    if side:
        check_hashes({str(Path(ready).parent/side['file']):side['sha256']})
    return r

def queue_decision(ready,complete,failure):
    if complete:return 'done'
    if ready:
        if failure and failure.get('status')=='training_failed_current_generation':return 'failed'
        return 'train'
    return 'blocked' if failure and failure.get('status')=='blocked_by_generator' else 'wait'

def require_delivery_manifest(path):
    path=Path(path)
    if not path.exists():raise ValueError('Missing completed-study stage manifest')
    r=json.loads(path.read_text(encoding='utf8'))
    if r.get('complete') is not True or r.get('pending') or r.get('stage_errors'):
        raise ValueError('Required comparison/audit stages are incomplete')
    from acceptance_manifest import expected_stage_keys,audit_coverage
    if set(r.get('coverage',{}))!=set(expected_stage_keys()):raise ValueError('Incomplete registered stage coverage')
    verified=audit_coverage(r['coverage'],expected_stage_keys(),r.get('artifacts',{}))
    if not verified['complete']:raise ValueError('Completed-study evidence changed or is incomplete')
    return r
