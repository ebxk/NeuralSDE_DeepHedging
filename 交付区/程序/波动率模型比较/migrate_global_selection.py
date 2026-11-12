"""Certify unchanged qualified global minima after the selector-only fix."""
import ast
import json
import shutil
from study_core import HERE,OUT,digest,write_json

def functions(source):return {n.name:n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}

def run():
    old=OUT/'attempts/pre-global-selector/stochastic_volatility.py'
    a=functions(old.read_text(encoding='utf8'));b=functions((HERE/'stochastic_volatility.py').read_text(encoding='utf8'))
    for key in a:
        if key!='fit_global' and ast.dump(a[key])!=ast.dump(b[key]):raise ValueError('Pricing/dynamic kernel changed')
    def prefix(n):
        result=[]
        for statement in n.body:
            result.append(ast.dump(statement))
            if isinstance(statement,ast.For):break
        return result
    if prefix(a['fit_global'])!=prefix(b['fit_global']):raise ValueError('Calibration preprocessing/optimizer changed')
    history=OUT/'attempts/global_selector_v1';history.mkdir(parents=True,exist_ok=True)
    results=[]
    for file in (OUT/'markets').glob('*/*/global.json'):
        r=json.loads(file.read_text(encoding='utf8'));identity=r['identity'];diag=r['model']['diagnostics']
        if identity['source']!=digest(old):continue
        if not diag.get('valid') or not diag.get('optimizer_success'):continue
        end,model=file.parent.parent.name,file.parent.name
        dest=history/f'{end}_{model}';dest.mkdir(exist_ok=True)
        for p in file.parent.iterdir():
            if p.is_file():shutil.copy2(p,dest/p.name)
        proof=dict(previous_receipt_sha256=digest(dest/file.name),
            statement='Old successful minimum passed all grids; unchanged optimization implies identical qualified minimum',
            previous_source_sha256=digest(old),current_source_sha256=digest(HERE/'stochastic_volatility.py'))
        r['identity']['source']=proof['current_source_sha256'];r['selector_migration']=proof;write_json(file,r)
        statusfile=file.parent/'status.json'
        status=json.loads(statusfile.read_text(encoding='utf8'))
        if not status['complete']:raise ValueError('Incomplete global model cannot be migrated')
        status['identity']=r['identity'];status['selector_migration']=proof
        status['output_sha256']={p.name:digest(p) for p in file.parent.iterdir() if p.is_file() and p.name!='status.json'}
        write_json(statusfile,status);results.append(dict(end=end,model=model,proof=proof))
    write_json(OUT/'global_selection_migration.json',dict(complete=True,migrated=results))

if __name__=='__main__':run()
