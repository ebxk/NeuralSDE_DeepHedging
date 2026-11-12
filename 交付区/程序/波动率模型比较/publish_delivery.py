"""Acceptance-gated independent main checkout; explicit aggregate allowlist."""
import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path,PurePosixPath,PureWindowsPath
import re
import shutil
import subprocess
import tempfile
from study_core import OUT,PROJECT,HERE,digest,write_json

REPOSITORY='https://github.com/ebxk/NeuralSDE_DeepHedging.git'
PREFIX='交付区/结果/波动率模型比较'
RESULTS={'seed_summary.csv','overall.csv','group_metrics.csv','paired_intervals.csv','historical_references.csv',
    'surface_comparison.csv','surface_status.json','direct_summary.csv','prediction_summary.csv','numerical_summary.csv',
    'numerical_diagnostics.csv','compute_costs.csv','numerical_status.json','简短结论.md','report_status.json',
    'completion.json','acceptance_summary.json','公开文件清单.json','comparison_figures.json',
    'year_stability_l0.png','year_stability_l0.1.png','pressure_main_l0.png','pressure_main_l0.1.png'}
PARAMS={'configuration.json','registry.json','requirements-lock.txt','requirements-market-lock.txt'}
EXCLUDED_PROGRAMS={'test_verify_predecessor.py','test_publish_delivery.py'}
ABS=re.compile(r'(?<![A-Za-z0-9_])[A-Za-z]:[\\/][^\s\"\'`<>)]*|/(?:Users|home|tmp|private|mnt|opt)/[^\s\"\'`<>)]*')
POSIX=re.compile(r'(?<![:/\w])/(?!/)[^\s\"\'`<>),;]+')
PRIVATE_COLUMNS={'episode','contract','code','date','origin','spot','strike','option_close','premium','pnl',
                 'price_observed','price_predicted','iv_observed','iv_predicted','ttm','k'}


def has_absolute_path(text):return bool(ABS.search(str(text)))


def _safe_string(value):
    value=ABS.sub(lambda match:PureWindowsPath(match.group()).name if re.match(r'^[A-Za-z]:',match.group())
                  else PurePosixPath(match.group()).name,value)
    return POSIX.sub(lambda match:PurePosixPath(match.group()).name,value)


def sanitize_json(value):
    if isinstance(value,list):return [sanitize_json(v) for v in value]
    if isinstance(value,dict):
        result={}
        for key,child in value.items():
            if key in ('local','executable','workspace','detailed_local_artifacts'):continue
            safe=_safe_string(str(key))
            if safe in result:safe+='__'+hashlib.sha256(str(key).encode()).hexdigest()[:8]
            result[safe]=sanitize_json(child)
        return result
    return _safe_string(value) if isinstance(value,str) else value


def allowed_relative(path):
    p=PurePosixPath(str(path).replace('\\','/'))
    if p.is_absolute() or '..' in p.parts or len(p.parts)!=4:return False
    head='/'.join(p.parts[:3]);name=p.name
    if head==PREFIX:return name in RESULTS or bool(re.fullmatch(r'tradeoffs_(2021|2022|2023|2025)\.png',name))
    if head=='交付区/参数与依赖/波动率模型比较':return name in PARAMS
    if head=='交付区/程序/波动率模型比较':return p.suffix in ('.py','.md','.txt') and name not in EXCLUDED_PROGRAMS
    return False


def validate_public_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as handle:
        reader=csv.DictReader(handle)
        if not reader.fieldnames or PRIVATE_COLUMNS&set(reader.fieldnames):raise ValueError('nonaggregate/raw CSV forbidden: '+str(path))
        for row in reader:
            if any(has_absolute_path(v) or _safe_string(v)!=v for v in row.values() if v):raise ValueError('absolute path inside aggregate CSV')


def portable_predecessor_helper(source):
    """Publish only unchanged generic comparison functions, not obsolete replay."""
    tree=ast.parse(source);keep={'EquivalenceError','_array_sha','assert_full_array_identity'}
    selected=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in keep]
    if {node.name for node in selected}!=keep:raise ValueError('portable helper extraction guard changed')
    lines=source.splitlines();body='\n\n'.join('\n'.join(lines[n.lineno-1:n.end_lineno]) for n in selected)
    result='"""Portable exact-array helpers; obsolete machine-specific replay CLI omitted."""\nimport hashlib\nimport json\nimport numpy as np\n\n'+body+'\n'
    rebuilt=ast.parse(result)
    original={n.name:ast.dump(n,include_attributes=False) for n in selected}
    public={n.name:ast.dump(n,include_attributes=False) for n in rebuilt.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    if original!=public:raise ValueError('generic financial comparison helper changed')
    return result


def public_text(name,text):
    if name=='verify_predecessor.py':text=portable_predecessor_helper(text)
    if name.endswith('.py'):
        if has_absolute_path(text):raise ValueError('program contains a machine path; explicit portable treatment required: '+name)
        return text
    return _safe_string(text)


def pin_dependencies(text,versions):
    lines=[]
    canonical=lambda name:re.sub(r'[-_.]+','-',name).lower()
    versions={canonical(k):v for k,v in versions.items()}
    for line in text.splitlines():
        if '@ file:' in line:
            name=line.split('@',1)[0].strip();version=versions.get(canonical(name))
            if not version:raise ValueError('cannot invent version for local-build requirement: '+name)
            line=name+'=='+version
        if has_absolute_path(line):raise ValueError('unresolved machine path in frozen dependency line')
        lines.append(line)
    return '\n'.join(lines)+'\n'


def _versions(interpreter):
    environment=dict(os.environ)
    for name in ('PYTHONHOME','PYTHONPATH','VIRTUAL_ENV'):environment.pop(name,None)
    code="import importlib.metadata as m,json;print(json.dumps({d.metadata['Name']:d.version for d in m.distributions()}))"
    return json.loads(subprocess.check_output([str(interpreter),'-c',code],env=environment,encoding='utf8'))


def _git(folder,*args):return subprocess.check_output(['git','-C',str(folder),*args],encoding='utf8').strip()


def publish(push=False):
    from artifact_guard import require_delivery_manifest
    acceptance=require_delivery_manifest(OUT/'acceptance_manifest.json')
    delivery=PROJECT/'交付区';result=delivery/'结果/波动率模型比较';params=delivery/'参数与依赖/波动率模型比较'
    completion=json.loads((result/'completion.json').read_text(encoding='utf8'))
    if completion.get('execution_complete') is not True:raise ValueError('local delivery is not finalized')
    local_summary=json.loads((result/'acceptance_summary.json').read_text(encoding='utf8'))
    if local_summary.get('manifest_sha256')!=digest(OUT/'acceptance_manifest.json'):raise ValueError('delivery acceptance generation changed')
    required=RESULTS-{'公开文件清单.json'}
    missing=sorted(name for name in required if not (result/name).is_file())
    if missing:raise ValueError('finalized aggregate publication package missing: '+','.join(missing))
    if _git(PROJECT,'remote','get-url','origin')!=REPOSITORY:raise ValueError('unexpected publication remote')
    temporary=PROJECT/'工作区/运行支持/临时文件';temporary.mkdir(parents=True,exist_ok=True)
    root=Path(tempfile.mkdtemp(prefix='volatility-public-',dir=temporary));bundle=root/'bundle';allowed=[];variants={};filtered=0
    def target(relative):
        relative=Path(relative)
        if not allowed_relative(relative):raise ValueError('not in publication whitelist: '+str(relative))
        path=bundle/relative;path.parent.mkdir(parents=True,exist_ok=True);allowed.append(relative.as_posix());return path
    for source in sorted((delivery/'程序/波动率模型比较').iterdir()):
        relative=Path('交付区/程序/波动率模型比较')/source.name
        if source.is_file() and allowed_relative(relative):
            text=source.read_text(encoding='utf-8-sig')
            if source.name.startswith('requirements'):
                runtime=PROJECT/('工作区/运行支持/环境/market/python.exe' if 'market' in source.name else '工作区/运行支持/环境/volatility/Scripts/python.exe')
                text=pin_dependencies(text,_versions(runtime))
            output=public_text(source.name,text);target(relative).write_text(output,encoding='utf8')
            if output!=text:variants[source.name]=dict(reason='portable generic helper or machine-path-redacted documentation',
                local_sha256=digest(source),public_sha256=digest(bundle/relative))
    for name in sorted(RESULTS-{'公开文件清单.json'}):
        source=result/name
        if not source.exists():continue
        relative=Path(PREFIX)/name;destination=target(relative)
        if source.suffix=='.json':destination.write_text(json.dumps(sanitize_json(json.loads(source.read_text(encoding='utf8'))),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
        elif source.suffix=='.csv':
            validate_public_csv(source)
            with source.open(encoding='utf-8-sig',newline='') as handle:
                reader=csv.DictReader(handle);fields=reader.fieldnames;rows=[]
                for row in reader:
                    n=row.get('n',row.get('paths',''))
                    if n and float(n)<2:filtered+=1;continue
                    rows.append(row)
            with destination.open('w',encoding='utf8',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        elif source.suffix=='.png':shutil.copy2(source,destination)
        else:destination.write_text(public_text(name,source.read_text(encoding='utf8')),encoding='utf8')
    for source in sorted(result.glob('tradeoffs_*.png')):shutil.copy2(source,target(Path(PREFIX)/source.name))
    for name in PARAMS:
        source=params/name;destination=target(Path('交付区/参数与依赖/波动率模型比较')/name)
        if source.suffix=='.json':text=json.dumps(sanitize_json(json.loads(source.read_text(encoding='utf8'))),ensure_ascii=False,indent=2,allow_nan=False)
        else:
            runtime=PROJECT/('工作区/运行支持/环境/market/python.exe' if 'market' in name else '工作区/运行支持/环境/volatility/Scripts/python.exe')
            text=pin_dependencies(source.read_text(encoding='utf-8-sig'),_versions(runtime))
        destination.write_text(text,encoding='utf8')
    # Validate every public textual byte before any fetch/clone/commit/push.
    for relative in allowed:
        p=bundle/relative
        if p.suffix!='.png' and has_absolute_path(p.read_text(encoding='utf8')):raise ValueError('absolute path survived public preparation: '+relative)
    checkout=root/'checkout'
    subprocess.run(['git','clone','--single-branch','--branch','main',REPOSITORY,str(checkout)],check=True)
    base=_git(checkout,'rev-parse','HEAD')
    for relative in allowed:
        p=checkout/relative;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(bundle/relative,p)
    manifest=dict(complete=True,remote_parent=base,registered=432,coverage_count=len(acceptance['coverage']),
        portable_source_variants=variants,omitted_nonessential_programs=sorted(EXCLUDED_PROGRAMS),small_cells_excluded=filtered,
        excluded=['raw quotes','per-path/daily data','npz arrays','model weights','machine paths'],
        sha256={name:digest(checkout/name) for name in allowed})
    manifest_relative=Path(PREFIX)/'公开文件清单.json';allowed.append(manifest_relative.as_posix())
    write_json(checkout/manifest_relative,manifest)
    _git(checkout,'-c','core.autocrlf=false','add','--',*allowed)
    staged=_git(checkout,'-c','core.quotepath=false','diff','--cached','--name-only').splitlines()
    if set(staged)-set(allowed) or not all(allowed_relative(Path(name)) for name in staged):raise ValueError('unexpected staged publication file')
    for name in staged:
        blob=subprocess.check_output(['git','-C',str(checkout),'show',':'+name])
        if hashlib.sha256(blob).hexdigest()!=digest(checkout/name):raise ValueError('staged bytes changed')
    if staged:_git(checkout,'-c','user.name=ebxk','-c','user.email=ebxk@users.noreply.github.com','commit','-m','Publish verified volatility comparison sources and aggregates')
    commit=_git(checkout,'rev-parse','HEAD')
    if push:
        _git(checkout,'push','origin','HEAD:refs/heads/main')
        if _git(checkout,'ls-remote','origin','refs/heads/main').split()[0]!=commit:raise ValueError('remote publication verification failed')
    report=dict(complete=True,published=bool(push),commit=commit,remote=REPOSITORY,remote_parent=base,files=len(allowed),checkout=str(checkout))
    write_json(OUT/'github_delivery.json',report);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--push',action='store_true');args=parser.parse_args();publish(args.push)

