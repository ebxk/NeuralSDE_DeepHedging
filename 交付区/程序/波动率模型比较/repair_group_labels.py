"""Correct holding-period metadata without changing any PnL or ledger value."""
import csv
import hashlib
import json
import zipfile
from datetime import datetime,timezone
import pandas as pd
from study_core import OUT,PROJECT,digest,write_json,bootstrap_legacy
from evaluate_common import crosses_ex_date
from report_common import risks

def run():
    bootstrap_legacy();from contract_ledger import EVENTS
    directory=OUT/'evaluation';overall=pd.read_csv(directory/'overall.csv');maps={};counts={};parents={}
    for end in sorted(overall.end.unique()):
        source=PROJECT/f'工作区/数据/处理后/创新组合实验/{end}/episode_days.csv';parents[str(source)]=digest(source)
        frame=pd.read_csv(source)
        dates=frame[frame.split_end==end].groupby('episode').date.agg(['min','max'])
        maps[end]={str(int(ep)):int(crosses_ex_date(r['min'],r['max'],EVENTS)) for ep,r in dates.iterrows()}
        counts[str(end)]=sum(maps[end].values())
    if counts['2025']!=77:raise ValueError('Expected 77 main holding-period dividend crossings')
    corrections=[]
    for record in overall.to_dict('records'):
        path=directory/f"{record['key']}_cost{int(record['cost_bp'])}_{record['execution']}.paths.csv"
        if not path.resolve().is_relative_to(directory.resolve()):raise ValueError('Path outside evaluation directory')
        with path.open(encoding='utf8',newline='') as handle:
            reader=csv.DictReader(handle);fields=reader.fieldnames;rows=list(reader)
        if len(rows)!=len(maps[record['end']]) or {r['episode'] for r in rows}!=set(maps[record['end']]):
            raise ValueError('Path-label denominator mismatch')
        changes=[i for i,r in enumerate(rows) if int(r['cross_dividend'])!=maps[record['end']][r['episode']]]
        if changes:corrections.append((path,fields,rows,changes,maps[record['end']]))
    if not corrections:
        groups=pd.read_csv(directory/'group_metrics.csv');groups=groups[groups.group=='cross_dividend']
        if all(int(row.n)==(counts[str(int(row.end))] if int(row.label) else len(maps[row.end])-counts[str(int(row.end))]) for row in groups.itertuples()):
            return dict(complete=True,already_correct=True,counts=counts)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    backup=OUT/f'attempts/cross_dividend_labels_{stamp}.zip';backup.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(backup,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path,*_ in corrections:archive.write(path,path.relative_to(OUT).as_posix())
        for name in ('group_metrics.csv','report_status.json'):archive.write(directory/name,'evaluation/'+name)
    evidence=[]
    for path,fields,rows,changes,flags in corrections:
        before=digest(path);protected=hashlib.sha256(json.dumps([{k:v for k,v in r.items() if k!='cross_dividend'} for r in rows]).encode()).hexdigest()
        for i in changes:rows[i]['cross_dividend']=str(flags[rows[i]['episode']])
        temporary=path.with_suffix('.csv.tmp')
        with temporary.open('w',encoding='utf8',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
        with temporary.open(encoding='utf8',newline='') as handle:after=list(csv.DictReader(handle))
        assert protected==hashlib.sha256(json.dumps([{k:v for k,v in r.items() if k!='cross_dividend'} for r in after]).encode()).hexdigest()
        temporary.replace(path);evidence.append(dict(file=path.name,changed_rows=len(changes),before=before,after=digest(path)))
    groups=directory/'group_metrics.csv'
    with groups.open(encoding='utf8',newline='') as handle:
        reader=csv.DictReader(handle);fields=reader.fieldnames;old=list(reader)
    result=[r for r in old if r['group']!='cross_dividend']
    for record in overall.to_dict('records'):
        path=directory/f"{record['key']}_cost{int(record['cost_bp'])}_{record['execution']}.paths.csv"
        frame=pd.read_csv(path,dtype={'episode':str})
        for flag,g in frame.groupby('cross_dividend'):
            rmse,cvar=risks(g.pnl.to_numpy());r={k:record[k] for k in fields if k in record}
            r.update(group='cross_dividend',label=int(flag),n=len(g),rmse=float(rmse),cvar=float(cvar),
                mean=float(g.pnl.mean()),cost=float(g.cost.mean()),turnover=float(g.turnover.mean()))
            result.append(r)
    if len(result)!=len(old):raise ValueError('Changed grouping coverage')
    temporary=groups.with_suffix('.csv.tmp')
    with temporary.open('w',encoding='utf8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(result)
    temporary.replace(groups)
    status_path=directory/'report_status.json';status=json.loads(status_path.read_text(encoding='utf8'))
    status['cross_dividend_rule']='entry_date < ex_date <= expiry';status['cross_dividend_counts']=counts;write_json(status_path,status)
    receipt=dict(complete=True,counts=counts,backup=str(backup),backup_sha256=digest(backup),
        parents=parents,
        files=evidence,pnl_cost_and_all_other_path_fields_unchanged=True,
        group_metrics_sha256=digest(groups),source=digest(__file__))
    write_json(directory/'group_label_repair.json',receipt);return receipt

if __name__=='__main__':
    result=run();print({k:v for k,v in result.items() if k!='files'})
