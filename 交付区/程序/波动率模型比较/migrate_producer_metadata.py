"""Prove the producer change only adds terminal receipt hashes, not model calculations."""
import ast
import copy
import json
import shutil
from study_core import HERE,OUT,digest,write_json

class WithoutStatusWrites(ast.NodeTransformer):
    def visit_Expr(self,node):
        if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='write_json':
            if 'status.json' in ast.unparse(node.value.args[0]):return None
        return self.generic_visit(node)

def run():
    old=OUT/'attempts/pre-producer-metadata/fit_stochastic.py';new=HERE/'fit_stochastic.py'
    left=WithoutStatusWrites().visit(ast.parse(old.read_text(encoding='utf8')))
    right=WithoutStatusWrites().visit(ast.parse(new.read_text(encoding='utf8')))
    if ast.dump(left)!=ast.dump(right):raise ValueError('Producer changed financial calculations')
    migrated=[]
    for status in (OUT/'markets').glob('*/*/status.json'):
        r=json.loads(status.read_text(encoding='utf8'))
        if not r.get('complete') or r.get('identity',{}).get('runner')!=digest(old):continue
        history=OUT/'attempts/producer_metadata_v1'/status.parent.parent.name/status.parent.name
        history.mkdir(parents=True,exist_ok=True)
        for p in status.parent.iterdir():
            if p.is_file():shutil.copy2(p,history/p.name)
        globalfile=status.parent/'global.json';g=json.loads(globalfile.read_text(encoding='utf8'))
        g['identity']['runner']=digest(new)
        g['producer_metadata_migration']=dict(old_runner=digest(old),new_runner=digest(new),financial_AST_identical=True)
        write_json(globalfile,g)
        r['identity']=g['identity'];r['producer_metadata_migration']=g['producer_metadata_migration']
        r['output_sha256']={name:digest(status.parent/name) for name in ('global.json','p_parameters.json','states.csv')}
        write_json(status,r);migrated.append(str(status))
    write_json(OUT/'producer_metadata_migration.json',dict(complete=True,migrated=migrated))

if __name__=='__main__':run()
