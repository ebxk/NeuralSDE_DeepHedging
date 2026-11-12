"""Preserve V1 receipts; certify unchanged valid minima, rerun rejected fits.

Only the selector changes: a previously selected converged, feasible, grid-valid
global minimum over feasible starts remains the minimum over qualified starts.
The optimization kernels and deterministic starting sequence must be identical.
"""
import ast
import json
import shutil
from pathlib import Path
from study_core import HERE,OUT,digest,write_json
from volatility_surfaces import _grid_check

def nodes(source):
    return {n.name:n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}

def run():
    archive=OUT/'attempts/pre-review-kernel';old=archive/'volatility_surfaces.py'
    before=old.read_text(encoding='utf8');after=(HERE/'volatility_surfaces.py').read_text(encoding='utf8')
    a,b=nodes(before),nodes(after)
    for name in a:
        if name!='fit_surface' and ast.dump(a[name])!=ast.dump(b[name]):
            raise ValueError(f'Financial kernel changed outside selector: {name}')
    old_fit=ast.get_source_segment(before,a['fit_surface']);new_fit=ast.get_source_segment(after,b['fit_surface'])
    if old_fit.split('    candidates = []')[0]!=new_fit.split('    candidates = []')[0]:
        raise ValueError('Calibration initialization/preprocessing changed')
    calls=lambda n:[ast.dump(x) for x in ast.walk(n) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='minimize']
    if calls(a['fit_surface'])!=calls(b['fit_surface']):raise ValueError('Optimizer call changed')
    old_runner=archive/'source/run_surfaces.py'
    if ast.dump(nodes(old_runner.read_text(encoding='utf8'))['calibrate'])!=ast.dump(nodes((HERE/'run_surfaces.py').read_text(encoding='utf8'))['calibrate']):
        raise ValueError('Calibration runner changed its financial computation')
    history=OUT/'attempts/surface_selection_v1';history.mkdir(parents=True,exist_ok=True)
    migrated=[];rerun=[]
    for path in (OUT/'surfaces').glob('*/*/*.json'):
        r=json.loads(path.read_text(encoding='utf8'))
        if r['identity']['source']==digest(HERE/'volatility_surfaces.py'):continue
        if r['identity']['source']!=digest(old):raise ValueError('Unknown old calibration kernel')
        relative=path.relative_to(OUT/'surfaces');dest=history/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest)
        d=r.get('model',{}).get('diagnostics',{})
        qualifies=r.get('status')=='complete' and d.get('optimizer_success') and d.get('constraint_min',-1)>=-1e-7 and _grid_check(r['model'],[.5,1.5])['valid']
        if qualifies:
            model_sha=__import__('hashlib').sha256(json.dumps(r['model'],sort_keys=True).encode()).hexdigest()
            r['identity']['source']=digest(HERE/'volatility_surfaces.py');r['identity']['runner']=digest(HERE/'run_surfaces.py')
            r['selector_migration']=dict(previous_receipt_sha256=digest(dest),model_unchanged_sha256=model_sha,
                proof='deterministic optimizer unchanged; previously qualified feasible minimum remains qualified minimum')
            write_json(path,r);migrated.append(str(relative))
        else:
            # The full V1 result remains in history; deletion is limited to its active cache alias.
            path.unlink();rerun.append(str(relative))
    write_json(OUT/'surface_selection_migration.json',dict(complete=True,old_source_sha256=digest(old),
        new_source_sha256=digest(HERE/'volatility_surfaces.py'),migrated=migrated,rerun=rerun))

if __name__=='__main__':run()
