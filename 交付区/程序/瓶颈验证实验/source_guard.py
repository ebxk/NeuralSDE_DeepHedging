"""Reject changes to inherited executable dependencies before reusing any jobs."""
import json
from pathlib import Path
from prepare_common import digest,write_json

def freeze(project,out):
    programs=project/'工作区/当前研究/程序'
    sources={}
    for folder in ('完整认购','创新组合实验','扩展组合实验'):
        for p in sorted((programs/folder).glob('*.py')):
            sources[str(p.relative_to(programs))]=digest(p)
    marker=out/'inherited_source_manifest.json'
    if marker.exists() and json.loads(marker.read_text(encoding='utf8'))!=sources:
        raise ValueError('Inherited source changed; preserve current study and create a new version')
    if not marker.exists():write_json(marker,sources)
    return sources
