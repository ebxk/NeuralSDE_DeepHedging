"""Read-only, fail-closed acceptance of the fixed approved comparison scope."""
import itertools
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from study_core import OUT,PROJECT,HERE,CONFIG,digest,write_json
from direct_failure import known_direct_rejection

ENDS=(2025,2021,2022,2023)
FOLDS={2025:[2022,2023,2025],2021:[2019,2020,2021],2022:[2020,2021,2022],2023:[2021,2022,2023]}
MODELS=('gbm','nsde','original_linear','svi_linear','ssvi_linear','essvi_linear','heston','localvol','sabr')
REPS=('original','nsde','svi','ssvi','essvi')
CASES={(0,'same_close'),(5,'same_close'),(10,'same_close'),(5,'next_close')}


def _jobs():
    return [dict(key=f'{end}_{model}_{info}_l{lam:g}_s{seed}',end=end,model=model,
                 information=info,lam=lam,seed=seed,initial_key=f'{end}_s{seed}',fold=FOLDS[end])
            for end,model,info,lam,seed in itertools.product(ENDS,MODELS,('base','surface'),(0.,.1),(0,1,2))]


def expected_stage_keys():
    keys=['data.preparation','data.clean_base','registry','surfaces','surface_report','numerical_report','account_acceptance','prediction.at_least_one_completed']
    for end in ENDS:
        keys.append(f'base_encoder.{end}')
        keys.append(f'reference.{end}')
        keys.extend(f'generator.{end}.{model}' for model in MODELS)
        keys.extend([f'direct.static.{end}',f'direct.nsde.{end}'])
        keys.extend(f'direct.dynamic.{end}.{model}' for model in ('heston','sabr','localvol'))
        keys.extend(f'factor.{end}.{rep}' for rep in REPS)
        keys.extend(f'prediction.{end}.{model}' for model in MODELS)
    keys.extend(f'policy.{job["key"]}' for job in _jobs())
    keys.extend(f'evaluation.job.{job["key"]}' for job in _jobs())
    keys.extend('evaluation.'+name for name in ('selection','overall','seed_summary','groups','paired','historical','report_status','nextclose'))
    return tuple(keys)


def audit_coverage(coverage,required,artifacts,stage_errors=None):
    """No empty/subset manifest, silent missing stage, or unlinked failure."""
    required=tuple(required);errors=list(stage_errors or []);pending=[];failures=[]
    if not required:errors.append(dict(stage='scope',reason='empty required coverage'))
    if set(coverage)-set(required):errors.append(dict(stage='scope',reason='unexpected coverage keys'))
    for key in required:
        row=coverage.get(key)
        if not row or row.get('status')=='pending':pending.append(key);continue
        if row.get('status')=='error':
            errors.append(dict(stage=key,reason=row.get('reason','stage integrity error')));continue
        if row.get('status') not in ('complete','declared_failure') or not row.get('artifacts'):
            errors.append(dict(stage=key,reason='resolved coverage requires evidence artifacts'));continue
        if row['status']=='declared_failure':
            kind=row.get('failure_kind');dependency=row.get('blocked_by')
            if not row.get('reason') or kind not in ('generator_blocked','quantified_fit_failure','training_failed_current_generation','blocked_by_generator','blocked_by_policy'):
                errors.append(dict(stage=key,reason='failure is not an executed scientific result'))
            if kind in ('blocked_by_generator','blocked_by_policy') and (
                not dependency or coverage.get(dependency,{}).get('status')!='declared_failure'):
                errors.append(dict(stage=key,reason='missing verified blocking dependency'))
            if kind in ('generator_blocked','training_failed_current_generation') and row.get('category')!='numerical_rejection':
                errors.append(dict(stage=key,reason='infrastructure or unclassified failure cannot close coverage'))
            seen={key};cursor=row
            while cursor.get('blocked_by'):
                parent=cursor['blocked_by']
                if parent in seen:
                    errors.append(dict(stage=key,reason='cyclic blocked coverage'));break
                seen.add(parent);cursor=coverage.get(parent,{})
            failures.append(dict(stage=key,**row))
        for path,sha in row['artifacts'].items():
            if artifacts.get(path)!=sha:errors.append(dict(stage=key,reason='coverage evidence absent from artifact set: '+path))
    for path,sha in artifacts.items():
        try:
            if not Path(path).is_file() or digest(path)!=sha:errors.append(dict(stage='artifact',reason='missing or changed bytes: '+path))
        except OSError as error:errors.append(dict(stage='artifact',reason=str(error)))
    return dict(complete=bool(required and not pending and not errors),pending=pending,stage_errors=errors,
                declared_failures=failures,coverage=coverage,artifacts=artifacts)


class Missing(Exception):pass


class Auditor:
    def __init__(self):
        self.coverage={};self.artifacts={};self.hashes={};self.current=[]
        self.jobs=_jobs();self.generator_states={};self.policy_states={};self.path_cache={};self.predictions=0
        self.cross_cache={}

    def artifact(self,path,expected=None):
        path=Path(path).resolve()
        if not path.is_file():raise Missing(str(path))
        stat=path.stat();version=(str(path),stat.st_mtime_ns,stat.st_size)
        if version not in self.hashes:self.hashes[version]=digest(path)
        sha=self.hashes[version]
        if expected is not None and sha!=expected:raise ValueError('hash mismatch: '+str(path))
        self.artifacts[str(path)]=sha;self.current.append(str(path));return sha

    def mappings(self,value,base):
        if isinstance(value,list):
            for child in value:self.mappings(child,base)
        if not isinstance(value,dict):return
        for name,child in value.items():
            if name in ('parents','sources','output_sha256','raw_sha256') and isinstance(child,dict):
                for file,sha in child.items():
                    if not isinstance(sha,str) or len(sha)!=64:continue
                    path=Path(file)
                    if not path.is_absolute():
                        path=(base/path if name=='output_sha256' else
                              PROJECT/'工作区/数据/原始数据/50ETF'/path if name=='raw_sha256' else HERE/path)
                    try:self.artifact(path,sha)
                    except Missing as error:raise ValueError('declared hash parent missing: '+str(error))
            elif isinstance(child,(dict,list)):self.mappings(child,base)

    def read(self,path):
        self.artifact(path);value=json.loads(Path(path).read_text(encoding='utf-8-sig'))
        self.mappings(value,Path(path).parent);return value

    def source(self,record,filename):
        identity=record.get('identity',record)
        source=identity.get('source',record.get('source_sha256'))
        if source:self.artifact(HERE/filename,source)

    def table(self,path):
        self.artifact(path)
        try:return pd.read_csv(path)
        except pd.errors.EmptyDataError:raise ValueError('empty required result table: '+str(path))

    def stage(self,key,callback):
        self.current=[]
        try:
            extra=callback() or {};row=dict(status='complete');row.update(extra)
        except Missing as error:row=dict(status='pending',reason=str(error))
        except Exception as error:row=dict(status='error',reason=f'{type(error).__name__}: {error}')
        row['artifacts']={p:self.artifacts[p] for p in dict.fromkeys(self.current)}
        self.coverage[key]=row;return row

    def paths(self,end):
        if end not in self.path_cache:
            path=PROJECT/f'工作区/数据/处理后/创新组合实验/{end}/episode_days.csv'
            frame=self.table(path);ids=frame.loc[frame.split_end==end,'episode'].drop_duplicates().astype(int).to_numpy()
            if not len(ids):raise ValueError('empty real-path partition')
            self.path_cache[end]=ids
            from evaluate_common import crosses_ex_date
            from study_core import bootstrap_legacy
            bootstrap_legacy();from contract_ledger import EVENTS
            dates=frame[frame.split_end==end].groupby('episode').date.agg(['min','max'])
            self.cross_cache[end]={int(ep):int(crosses_ex_date(row['min'],row['max'],EVENTS)) for ep,row in dates.iterrows()}
            if end==2025 and sum(self.cross_cache[end].values())!=77:raise ValueError('Main holding-period cross-dividend paths must be 77')
        else:self.artifact(PROJECT/f'工作区/数据/处理后/创新组合实验/{end}/episode_days.csv')
        return self.path_cache[end]

    def blocked(self,end,model):
        ready=OUT/f'inputs/{end}/sim_{model}.ready.json'
        if ready.exists():raise ValueError('existing readiness cannot be superseded by old blocked records')
        records=[]
        for job in self.jobs:
            if job['end']==end and job['model']==model:
                record=self.read(OUT/f'failures/{job["key"]}.json')
                if record.get('status')!='blocked_by_generator' or record.get('category')!='numerical_rejection' or not record.get('reason'):
                    raise ValueError('generator block is missing, old, or infrastructure failure')
                if any(record.get('job',{}).get(k)!=v for k,v in job.items()):raise ValueError('blocked job identity mismatch')
                records.append(record)
        evidence=False
        global_status=OUT/f'markets/{end}/{model}/status.json'
        if model in ('heston','sabr') and global_status.exists():
            status=self.read(global_status)
            if status.get('status')=='global_numerical_rejected':
                global_fit=self.read(global_status.parent/'global.json');self.source(global_fit,'stochastic_volatility.py')
                if global_fit.get('identity',{}).get('runner'):self.artifact(HERE/'fit_stochastic.py',global_fit['identity']['runner'])
                if global_fit.get('identity',{}).get('quotes'):self.artifact(OUT/'data/quotes.csv',global_fit['identity']['quotes'])
                evidence=global_fit.get('model',{}).get('diagnostics',{}).get('valid') is False
        failure=OUT/f'inputs/{end}/sim_{model}.failure.json'
        if failure.exists():
            value=self.read(failure)
            if value.get('status')=='numerical_failed' and value.get('model')==model and value.get('end')==end:
                # A current producer source is mandatory for rejecting a branch.
                source=value.get('source',value.get('source_sha256'))
                if source:self.artifact(HERE/'generator_export.py',source);evidence=True
        for record in records:
            producer=record.get('producer_evidence')
            if producer and isinstance(producer,dict):
                self.artifact(producer['path'],producer['sha256']);decl=self.read(producer['path'])
                if decl.get('status') in ('numerical_failed','global_numerical_rejected'):
                    source=decl.get('source',decl.get('source_sha256'))
                    if source:self.artifact(HERE/producer.get('source_file','generator_export.py'),source);evidence=True
        if not evidence:raise ValueError('blocked records lack current-source numerical producer evidence')
        return dict(status='declared_failure',failure_kind='generator_blocked',category='numerical_rejection',reason='current producer numerical rejection',resolved_jobs=len(records))

    def generator(self,end,model):
        path=OUT/f'inputs/{end}/sim_{model}.ready.json'
        if not path.exists():return self.blocked(end,model)
        r=self.read(path)
        if not r.get('parents',r.get('identity',{}).get('parents')):raise ValueError('generator has no parent provenance')
        parents=r.get('parents',r.get('identity',{}).get('parents',{}))
        base=OUT/f'representations/{end}/original.npz'
        if str(base) not in parents and str(base.resolve()) not in parents:raise ValueError('generator lacks current clean-base encoder parent')
        self.source(r,'common_inputs.py' if model=='gbm' else 'generator_export.py')
        from artifact_guard import verify_ready
        verify_ready(path,path.with_name(f'sim_{model}.npz'),CONFIG,model,end)
        self.artifact(path.with_name(f'sim_{model}.npz'),r['input_sha256'])
        if r.get('identity_sha256'):self.artifact(path.parent/'identity.json',r['identity_sha256'])
        side=r.get('state_sidecar')
        if model!='gbm' and not side:raise ValueError('required latent/factor sidecar missing')
        if side:
            if side.get('complete') is not True:raise ValueError('state sidecar is not complete')
            self.artifact(path.parent/side['file'],side['sha256'])
        with np.load(path.with_name(f'sim_{model}.npz'),allow_pickle=False) as data:
            if data['spot'].shape[0]!=40960 or data['f'].shape[-1]!=7:raise ValueError('generator full budget/features mismatch')
        return dict(rows=40960)

    def dependency(self,key,kind='blocked_by_generator'):
        row=self.coverage.get(key,{})
        if row.get('status')!='declared_failure':raise Missing('unresolved dependency '+key)
        for path,sha in row['artifacts'].items():self.artifact(path,sha)
        return dict(status='declared_failure',failure_kind=kind,blocked_by=key,reason='explicit verified dependency failure')

    def policy(self,job):
        key=job['key'];generator=f'generator.{job["end"]}.{job["model"]}'
        if self.coverage.get(generator,{}).get('status')=='declared_failure':return self.dependency(generator)
        if self.coverage.get(generator,{}).get('status')!='complete':raise Missing(generator)
        path=OUT/f'policies/{key}.json'
        if path.exists():
            r=self.read(path);identity=r['identity']
            if r.get('status')!='trained_validation_selected' or identity.get('config')!=json.loads(json.dumps(CONFIG)) or not identity.get('parents'):
                raise ValueError('invalid policy training receipt/config')
            actual=identity['job']
            if any(actual.get(k)!=v for k,v in job.items()):raise ValueError('policy registry identity mismatch')
            self.artifact(path.with_suffix('.pt'),r['weight_sha256'])
            self.artifact(OUT/f'initial/{job["initial_key"]}.pt',identity['initial_sha256'])
            return dict(weight_sha256=r['weight_sha256'])
        r=self.read(OUT/f'failures/{key}.json')
        ready=OUT/f'inputs/{job["end"]}/sim_{job["model"]}.ready.json'
        context=dict(ready=self.artifact(ready),trainer=self.artifact(HERE/'train_common.py'),policy=self.artifact(HERE/'common_policy.py'))
        if r.get('status')!='training_failed_current_generation' or r.get('context')!=context or r.get('key')!=key:
            raise ValueError('unprocessed or stale-generation training failure')
        if r.get('category') not in ('numerical_rejection','numeric') or not r.get('reason'):
            raise ValueError('training infrastructure failure cannot close coverage')
        return dict(status='declared_failure',failure_kind='training_failed_current_generation',category='numerical_rejection',reason=r['reason'])

    def scenarios(self,path,end,method,methods):
        frame=self.table(path);expected=set(map(int,self.paths(end)))
        cost='cost_bp' if 'cost_bp' in frame else 'bp'
        expected_groups={(name,bp,execution) for name in methods for bp,execution in CASES}
        found=set()
        for key,g in frame.groupby([method,cost,'execution'],dropna=False):
            normalized=(key[0],int(key[1]),key[2]);found.add(normalized)
            if len(g)!=len(expected) or set(map(int,g.episode))!=expected:raise ValueError('scenario path IDs/count mismatch')
        if found!=expected_groups:raise ValueError('incomplete scenario matrix')
        return frame

    def direct(self,end,kind):
        if kind=='static':
            directory=OUT/f'direct/{end}';r=self.read(directory/'status.json')
            if r.get('complete') is not True or r.get('scenario_groups')!=28:raise Missing('static 28-group completion')
            frame=self.table(directory/'path_pnl.csv');expected=set(map(int,self.paths(end)))
            definitions={('bs','black'),*itertools.product(('svi','ssvi','essvi'),('black','smile'))};found=set()
            for key,g in frame.groupby(['model','delta','cost_bp','execution']):
                found.add((key[0],key[1],int(key[2]),key[3]))
                if len(g)!=len(expected) or set(map(int,g.episode))!=expected:raise ValueError('static path denominator mismatch')
            if found!={(m,d,b,e) for m,d in definitions for b,e in CASES}:raise ValueError('static missing cost/execution groups')
            self.table(directory/'group_metrics.csv');self.artifact(directory/'ratios.csv');daily=directory/'daily_ledger.csv'
        elif kind=='nsde':
            directory=OUT/f'direct_nsde/{end}';r=self.read(directory/'status.json')
            if r.get('complete') is not True:raise Missing('NSDE direct benchmark unfinished')
            self.scenarios(directory/'pnl.csv',end,'method',('curve_delta','local_minimum_variance'));daily=directory/'daily_ledger.csv.gz'
            for field,path in [('factors_sha256',OUT/f'representations/{end}/original.npz'),
                               ('common_sha256',PROJECT/f'工作区/数据/处理后/创新组合实验/{end}/C1/common.npz'),
                               ('fitted_reference_sha256',OUT/f'markets/{end}/nsde/reference_common.npz'),
                               ('episodes_sha256',PROJECT/f'工作区/数据/处理后/创新组合实验/{end}/episode_days.csv')]:
                if field in r:self.artifact(path,r[field])
        else:
            directory=OUT/f'direct_dynamic/{end}/{kind}';r=self.read(directory/'status.json')
            if r.get('status')=='benchmark_missing':return self.dependency(f'generator.{end}.{kind}')
            if known_direct_rejection(r,end,kind):
                return dict(status='declared_failure',failure_kind='quantified_fit_failure',reason=r['error'])
            if r.get('complete') is not True:raise ValueError('dynamic direct numerical/infrastructure failure')
            self.scenarios(directory/'pnl.csv',end,'hedge',('delta',) if kind=='localvol' else ('delta','lmv'))
            self.table(directory/'groups.csv');daily=directory/'daily_ledger.csv.gz'
        if r.get('paths')!=len(self.paths(end)):raise ValueError('direct receipt path count mismatch')
        if 'path_ids_sha256' in r:
            sha=hashlib.sha256(json.dumps(self.paths(end).tolist(),separators=(',',':')).encode()).hexdigest()
            if r['path_ids_sha256']!=sha:raise ValueError('direct path identity hash mismatch')
        if r.get('episode_source_sha256'):self.artifact(PROJECT/f'工作区/数据/处理后/创新组合实验/{end}/episode_days.csv',r['episode_source_sha256'])
        self.artifact(daily)
        return dict(paths=len(self.paths(end)))

    def factor(self,end,rep):
        directory=OUT/f'factor_quote_audit/{end}/{rep}';r=self.read(directory/'status.json')
        if r.get('status')=='failed':
            if rep in ('svi','ssvi','essvi') and r.get('error_type')=='FileNotFoundError':
                return self.dependency(f'generator.{end}.{rep}_linear')
            raise ValueError('factor audit script failed, not a scientific fit rejection')
        if r.get('status') not in ('complete','fit_failed'):raise Missing('factor audit unfinished')
        frame=self.table(directory/'daily.csv');summary=self.read(directory/'summary.json');self.artifact(directory/'daily.jsonl')
        if set(frame.view)!= {'full','heldout'} or frame.duplicated(['date','view']).any():raise ValueError('factor view/date coverage invalid')
        quotes=self.table(OUT/'data/quotes.csv');dates=set(quotes.loc[quotes.date<str(end+1),'date'])
        if set(frame.date)!=dates or len(frame)!=2*len(dates):raise ValueError('factor missing full/heldout dates')
        if {row['view'] for row in summary}!={'full','heldout'}:raise ValueError('factor summary lacks masked/full coverage')
        identity=r.get('identity',{});self.source(r,'factor_quote_audit.py')
        if not identity.get('source') or not identity.get('parents') or identity.get('fold')!=FOLDS[end]:raise ValueError('factor provenance/fold missing')
        expected_groups=set(frame.groupby(['split','view']).groups)
        if len(summary)!=len(expected_groups) or {(s['split'],s['view']) for s in summary}!=expected_groups:
            raise ValueError('factor split/view summary incomplete')
        for s in summary:
            part=frame[(frame.split==s['split'])&(frame.view==s['view'])]
            if s['visible_n']!=int(part.visible_n.sum()) or s['heldout_n']!=int(part.heldout_n.sum()):raise ValueError('factor aggregate quote counts mismatch')
        if identity.get('initialization_sha256') and identity.get('starts') is not None:
            if hashlib.sha256(np.asarray(identity['starts'],float).tobytes()).hexdigest()!=identity['initialization_sha256']:
                raise ValueError('factor initial-state SHA mismatch')
        identity_file=self.read(directory/'identity.json')
        if identity_file!=identity:raise ValueError('factor identity receipt differs from saved state identity')
        identity_sha=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        if not frame.identity_sha256.eq(identity_sha).all():raise ValueError('factor daily identity hash mismatch')
        if r.get('status')=='fit_failed':
            if not r.get('failed_fits') or 'valid' not in frame or not (~frame.valid.astype(bool)).any():raise ValueError('unquantified factor failure')
            return dict(status='declared_failure',failure_kind='quantified_fit_failure',reason='saved independent masked fits include quantified failures')

    def prediction(self,end,model):
        path=OUT/f'predictions/{end}_{model}.json'
        dependency=f'generator.{end}.{model}'
        state=self.coverage.get(dependency,{}).get('status')
        if state=='declared_failure':
            if path.exists() and self.read(path).get('complete') is True:raise ValueError('completed prediction belongs to a currently blocked generator generation')
            return self.dependency(dependency)
        if state!='complete':raise Missing(dependency)
        if not path.exists():return self.dependency(f'generator.{end}.{model}')
        r=self.read(path)
        if r.get('complete') is not True:
            if r.get('status')=='blocked_by_generator':return self.dependency(f'generator.{end}.{model}')
            raise Missing('conditional prediction not complete')
        if (r.get('model')!=model or r.get('end')!=end or r.get('config')!=json.loads(json.dumps(CONFIG))
            or not r.get('source') or not r.get('parents') or not r.get('output_sha256')):
            raise ValueError('prediction source/input/output provenance missing or wrong generation')
        self.source(r,'prediction_audit.py')
        frame=self.table(path.with_suffix('.csv'))
        if r.get('results')!=len(frame) or frame.empty:raise ValueError('prediction has no inspectable numerical rows')
        if not set(frame.horizon).issubset({1,5,20}):raise ValueError('prediction horizon changed')
        if frame.duplicated(['origin','horizon']).any() or not (frame.n>=2).all():raise ValueError('prediction origin/sample rows malformed')
        if 'residual_status' not in frame or not frame.residual_status.eq('own_model_price_increments').all():raise ValueError('prediction does not certify own-model conditional prices')
        if not np.isfinite(frame[['residual_crps','residual_observed']].to_numpy()).all():raise ValueError('nonfinite conditional prediction scores')
        covered=set(frame.horizon)|{row['horizon'] for row in r.get('excluded',[])}
        if covered!={1,5,20}:raise ValueError('prediction horizons are neither computed nor explicitly excluded')
        self.predictions+=1
        return dict(results=len(frame))

    def evaluation_job(self,job):
        key=job['key'];dependency='policy.'+key
        if self.coverage.get(dependency,{}).get('status')=='declared_failure':return self.dependency(dependency,'blocked_by_policy')
        if self.coverage.get(dependency,{}).get('status')!='complete':raise Missing(dependency)
        frame=self.table(OUT/'evaluation/overall.csv');rows=frame[frame.key==key]
        if len(rows)!=4 or {(int(r.cost_bp),r.execution) for r in rows.itertuples()}!=CASES:raise ValueError('trained job missing one of four evaluation cases')
        expected=set(map(int,self.paths(job['end'])))
        for row in rows.itertuples():
            if int(row.n)!=len(expected) or not np.isfinite(row.ledger_max_error) or row.ledger_max_error>1e-8:raise ValueError('evaluation denominator or independent ledger error invalid')
            stem=OUT/f'evaluation/{key}_cost{int(row.cost_bp)}_{row.execution}'
            paths=self.table(Path(str(stem)+'.paths.csv'));self.artifact(Path(str(stem)+'.ledger.csv.gz'))
            if len(paths)!=len(expected) or set(map(int,paths.episode))!=expected:raise ValueError('evaluation path IDs differ by case')
            if any(int(flag)!=self.cross_cache[job['end']][int(ep)] for ep,flag in zip(paths.episode,paths.cross_dividend)):
                raise ValueError('Cross-dividend labels do not match the actual holding interval')
            self.artifact(Path(str(stem)+'.npz'))
            with np.load(Path(str(stem)+'.npz'),allow_pickle=False) as data:
                if len(data['pnl'])!=len(expected) or not np.isfinite(data['pnl']).all():raise ValueError('evaluation PnL array malformed')


def build_manifest():
    a=Auditor()
    a.artifact(HERE/'acceptance_manifest.py')
    def preparation():
        r=a.read(OUT/'data/preparation.json')
        if r.get('complete') is not True or r.get('dates')!=2646:raise Missing('full 2646-date data preparation')
        quotes=a.table(OUT/'data/quotes.csv');a.table(OUT/'data/quote_exclusions.csv')
        if len(quotes)!=r.get('accepted_quotes') or quotes.date.nunique()!=2646:raise ValueError('prepared quote counts mismatch')
        if quotes.duplicated(['date','expire_date','strike']).any():raise ValueError('prepared duplicate effective quotes')
        for end in ENDS:
            episode=a.read(OUT/f'data/episodes_{end}.json')
            if episode.get('sha256'):a.artifact(PROJECT/f'工作区/数据/处理后/创新组合实验/{end}/episode_days.csv',episode['sha256'])
    def registry():
        r=a.read(OUT/'registry.json');jobs=r.get('jobs',[])
        if len(jobs)!=432 or len({j['key'] for j in jobs})!=432 or {j['key'] for j in jobs}!={j['key'] for j in a.jobs}:raise ValueError('registry does not have exactly approved 432 unique jobs')
        approved={j['key']:j for j in a.jobs}
        if any(any(job.get(k)!=v for k,v in approved[job['key']].items()) for job in jobs):raise ValueError('registry job coordinates drifted')
        if r.get('config')!=json.loads(json.dumps(CONFIG)):raise ValueError('registry configuration drift')
    def clean_base():
        r=a.read(OUT/'data/clean_base/status.json');a.source(r,'prepare_clean_base.py')
        if r.get('complete') is not True:raise Missing('shared clean quote baseline unfinished')
        if not r.get('source',r.get('identity',{}).get('source')) or not r.get('parents',r.get('identity',{}).get('parents')):
            raise ValueError('clean-base provenance missing')
        a.artifact(OUT/'data/quotes.csv',r['quote_sha256'])
        a.artifact(OUT/'data/shared_surfaces.npz',r['shared_surfaces_sha256'])
        with np.load(OUT/'data/shared_surfaces.npz',allow_pickle=False) as data:
            if len(data['dates'])!=2646 or data['calls'].shape!=(2646,20):raise ValueError('shared clean baseline coverage mismatch')
            if float(np.min(data['calls']@data['A'].T-data['b']))<-1e-7:raise ValueError('shared clean baseline financial infeasibility')
    def surfaces():
        r=a.read(OUT/'surface_stage.json')
        if r.get('stage_complete') is not True:raise Missing('full surface stage is not complete')
        quotes=a.table(OUT/'data/quotes.csv');dates=sorted(quotes.date.unique())
        expected={(day,kind,view) for day in dates for kind in ('svi','ssvi','essvi') for view in ('full','heldout')}
        results=r.get('results',[])
        if len(dates)!=2646 or r.get('expected')!=15876 or len(results)!=15876:raise ValueError('surface stage must record 2646*6 results')
        if {(row['day'],row['kind'],row['view']) for row in results}!=expected:raise ValueError('surface result identity coverage differs')
        failed=0
        for day,kind,view in sorted(expected):
            record=a.read(OUT/f'surfaces/{kind}/{day}/{view}.json');a.source(record,'volatility_surfaces.py')
            identity=record['identity']
            if identity.get('runner'):a.artifact(HERE/'run_surfaces.py',identity['runner'])
            if identity.get('quotes'):a.artifact(OUT/'data/quotes.csv',identity['quotes'])
            if record.get('status') not in ('complete','numerical_rejected'):raise ValueError('surface runtime failure is not a scientific rejection')
            if record['status']=='complete' and not record.get('model',{}).get('diagnostics',{}).get('valid'):raise ValueError('surface complete but numerical validity false')
            if record['status']=='numerical_rejected' and record.get('model',{}).get('diagnostics',{}).get('valid') is not False:raise ValueError('unquantified numerical surface rejection')
            failed+=record['status']=='numerical_rejected'
        if failed:return dict(status='declared_failure',failure_kind='quantified_fit_failure',reason=f'{failed} saved numerical surface rejections',results=15876)
        return dict(results=15876)
    def account():
        r=a.read(OUT/'account_acceptance.json')
        if r.get('complete') is not True or r.get('paths')!=773:raise Missing('773-path account acceptance')
        errors=[r['independent_ledger_error'],r['legacy_reload_error']]
        if not np.isfinite(errors).all() or max(errors)>1e-8:raise ValueError('account parity exceeds 1e-8')
        if not r.get('includes_dividends_adjustments_and_next_close'):raise ValueError('account acceptance lacks required mechanics')
        if r.get('legacy_weight_sha256'):a.artifact(PROJECT/'工作区/数据/处理后/创新组合实验/2025/policies/C1_single_5bp_l0_s0/same_capacity.pt',r['legacy_weight_sha256'])
    a.stage('data.preparation',preparation);a.stage('data.clean_base',clean_base);a.stage('registry',registry);a.stage('surfaces',surfaces);a.stage('account_acceptance',account)
    for end in ENDS:
        def base_encoder(end=end):
            directory=OUT/f'representations/{end}';r=a.read(directory/'original.json');a.source(r,'prepare_clean_base.py')
            if r.get('complete') is not True:raise Missing('clean original encoder unfinished')
            if r.get('train_end')!=FOLDS[end][0]:raise ValueError('encoder training cutoff mismatch')
            if not r.get('source',r.get('identity',{}).get('source')) or not r.get('parents',r.get('identity',{}).get('parents')):raise ValueError('encoder provenance missing')
            a.artifact(OUT/'data/shared_surfaces.npz',r['shared_surfaces_sha256']);a.artifact(directory/'original.npz')
            with np.load(directory/'original.npz',allow_pickle=False) as rep:
                if len(rep['dates'])!=2646 or rep['basis'].shape!=(2,20):raise ValueError('clean encoder geometry/coverage mismatch')
                if float(np.min(rep['xi']@rep['W'].T-rep['bounds']))<-1e-7:raise ValueError('clean encoder factor states infeasible')
        a.stage(f'base_encoder.{end}',base_encoder)
        def reference(end=end):
            directory=OUT/f'markets/{end}/nsde';r=a.read(directory/'market.json');a.source(r,'reference_market.py')
            if r.get('complete') is not True:raise Missing('fresh nonlinear NSDE reference unavailable')
            identity=r['identity']
            if not identity.get('parents') or not identity.get('source') or identity.get('end')!=end or identity.get('study_config')!=json.loads(json.dumps(CONFIG)):
                raise ValueError('fresh reference current-generation provenance/config incomplete')
            if identity.get('fit_config',{}).get('split_ends')!=FOLDS[end]:raise ValueError('fresh reference fold mismatch')
            a.artifact(OUT/f'representations/{end}/original.npz',identity['factor'])
            for field,path in [('episodes',PROJECT/f'工作区/数据/处理后/创新组合实验/{end}/episode_days.csv'),
                               ('config',PROJECT/f'工作区/数据/处理后/创新组合实验/{end}/config.json'),
                               ('rawspot',PROJECT/'工作区/数据/原始数据/50ETF/510050_华夏上证50ETF.csv')]:
                if field in identity:a.artifact(path,identity[field])
            if identity.get('head_widths')!=[64,64,7] or identity.get('market_weights_reused') is not False:raise ValueError('reference is not freshly fitted nonlinear NSDE')
            if identity.get('actual_original_market_source_sha256'):a.artifact(PROJECT/'工作区/当前研究/程序/创新组合实验/market.py',identity['actual_original_market_source_sha256'])
            for name in ('simulation.npz','reference_common.npz','market_2026.weights.h5'):a.artifact(directory/name)
        a.stage(f'reference.{end}',reference)
        for model in MODELS:a.stage(f'generator.{end}.{model}',lambda end=end,model=model:a.generator(end,model))
    for job in a.jobs:a.stage('policy.'+job['key'],lambda job=job:a.policy(job))
    for end in ENDS:
        for kind in ('static','nsde','heston','sabr','localvol'):
            key=f'direct.{kind}.{end}' if kind in ('static','nsde') else f'direct.dynamic.{end}.{kind}'
            a.stage(key,lambda end=end,kind=kind:a.direct(end,kind))
        for rep in REPS:a.stage(f'factor.{end}.{rep}',lambda end=end,rep=rep:a.factor(end,rep))
        for model in MODELS:a.stage(f'prediction.{end}.{model}',lambda end=end,model=model:a.prediction(end,model))
    def any_prediction():
        if not a.predictions:raise Missing('no completed conditional prediction artifacts; an entirely absent prediction stage cannot close')
        for end in ENDS:
            for model in MODELS:
                if a.coverage[f'prediction.{end}.{model}']['status']=='complete':a.artifact(OUT/f'predictions/{end}_{model}.json')
        return dict(completed_predictions=a.predictions)
    a.stage('prediction.at_least_one_completed',any_prediction)
    for job in a.jobs:a.stage('evaluation.job.'+job['key'],lambda job=job:a.evaluation_job(job))
    def selection():
        r=a.read(OUT/'selected_checkpoints.json');trained={j['key'] for j in a.jobs if a.coverage['policy.'+j['key']]['status']=='complete'}
        if set(r.get('selections',{}))!=trained:raise ValueError('selected checkpoint registry differs from trained policies')
        unavailable={j['key'] for j in a.jobs}-trained
        if {j['key'] for j in r.get('unavailable',[])}!=unavailable:raise ValueError('unavailable checkpoint partition differs from explicit failures')
        for key,sha in r['selections'].items():a.artifact(OUT/f'policies/{key}.json',sha)
    a.stage('evaluation.selection',selection)
    table_files=dict(overall='overall.csv',seed_summary='seed_summary.csv',groups='group_metrics.csv',paired='paired_intervals.csv',historical='historical_references.csv')
    for key,name in table_files.items():
        def table_check(key=key,name=name):
            frame=a.table(OUT/'evaluation'/name)
            if frame.empty:raise ValueError('required evaluation result table is empty: '+name)
            overall=a.table(OUT/'evaluation/overall.csv')
            trained={j['key'] for j in a.jobs if a.coverage['policy.'+j['key']]['status']=='complete'}
            if key=='historical':
                expected=set(itertools.product(ENDS,('legacy_C0','legacy_C1'),(0,1,2)))
                if len(frame)!=24 or set(zip(frame.end,frame.kind,frame.seed))!=expected:raise ValueError('historical four-fold/C0/C1/seed references incomplete')
                for row in frame.itertuples():
                    source=PROJECT/f'工作区/数据/处理后/瓶颈验证实验/evaluation/paths/{int(row.end)}_{row.kind}_s{int(row.seed)}_5bp_delay0.json'
                    a.artifact(source,row.source_sha256)
            elif key=='overall':
                if set(frame.key)!=trained or len(frame)!=4*len(trained):raise ValueError('overall does not match exactly all trained job cases')
                if not np.isfinite(frame[['rmse','cvar','mean','std','cost','turnover','ledger_max_error']].to_numpy()).all():raise ValueError('nonfinite overall metrics')
            elif key=='seed_summary':
                cols=['end','model','information','lam','cost_bp','execution']
                expected=set(overall[cols].itertuples(index=False,name=None))
                if len(frame)!=len(expected) or set(frame[cols].itertuples(index=False,name=None))!=expected:raise ValueError('seed summary missing registered comparison groups')
            elif key=='groups':
                dims={'term','moneyness','month','cross_dividend','adjusted'}
                expected={(r.key,int(r.cost_bp),r.execution,d) for r in overall.itertuples() for d in dims}
                found=set()
                for coords,g in frame.groupby(['key','cost_bp','execution','group']):
                    found.add((coords[0],int(coords[1]),coords[2],coords[3]))
                    end=int(g.end.iloc[0])
                    if int(g.n.sum())!=len(a.paths(end)):raise ValueError('group labels do not partition common paths')
                if found!=expected:raise ValueError('grouping dimensions/cost cases incomplete')
            elif key=='paired':
                cols=['end','model','information','lam','cost_bp','execution']
                by={k:set(g.seed) for k,g in overall.groupby(cols)};expected=set()
                for coords,seeds in by.items():
                    baseline=(coords[0],'nsde',*coords[2:])
                    if coords[1]!='nsde' and seeds=={0,1,2} and by.get(baseline)=={0,1,2}:
                        expected.update((*coords,block) for block in (1,3,6))
                actual=set(frame[cols+['block_months']].itertuples(index=False,name=None))
                if len(frame)!=len(expected) or actual!=expected:raise ValueError('paired month-block intervals missing comparisons or blocks')
                if not frame.repeats.eq(2000).all():raise ValueError('paired bootstrap budget changed')
            if key in ('overall','seed_summary','groups'):
                if set(frame.end)!=set(ENDS):raise ValueError('evaluation table lacks one or more folds')
            return dict(rows=len(frame))
        a.stage('evaluation.'+key,table_check)
    def report():
        r=a.read(OUT/'evaluation/report_status.json')
        if r.get('tables_complete') is not True:raise Missing('final report tables not complete')
        for field,name in [('seed_rows','seed_summary.csv'),('group_rows','group_metrics.csv'),('interval_rows','paired_intervals.csv')]:
            if r.get(field)!=len(a.table(OUT/'evaluation'/name)):raise ValueError('report completion counts mismatch')
        for end in ENDS:a.artifact(OUT/f'evaluation/tradeoffs_{end}.png')
    def nextclose():
        frame=a.table(OUT/'evaluation/overall.csv');trained={j['key'] for j in a.jobs if a.coverage['policy.'+j['key']]['status']=='complete'}
        rows=frame[(frame.execution=='next_close')&(frame.cost_bp==5)]
        if set(rows.key)!=trained or len(rows)!=len(trained):raise ValueError('next-close coverage does not match every trained job')
    a.stage('evaluation.report_status',report);a.stage('evaluation.nextclose',nextclose)
    def surface_report():
        r=a.read(OUT/'evaluation/surface_status.json');a.source(r,'surface_report.py')
        if r.get('complete') is not True or r.get('source_errors'):raise Missing('static price/IV reconstruction report incomplete')
        from surface_report import METHODS,REGIONS
        if set(r.get('methods',[]))!=set(METHODS):raise ValueError('static comparison method scope mismatch')
        frame=a.table(OUT/'evaluation/surface_comparison.csv')
        expected=set(itertools.product(ENDS,METHODS,('training','validation','historical_evaluation'),('full','heldout'),REGIONS))
        coords=['end','method','split','view','region']
        if len(frame)!=len(expected) or set(frame[coords].itertuples(index=False,name=None))!=expected:
            raise ValueError('static comparison fold/method/view/domain matrix incomplete')
        required=['price_rmse','iv_rmse','matched_price_rmse','matched_iv_rmse','low_vega_n','unidentifiable_n']
        if not set(required).issubset(frame.columns):raise ValueError('static price/IV/identifiability metrics missing')
        a.artifact(OUT/'evaluation/surface_daily.csv');a.artifact(OUT/'evaluation/quote_errors.csv.gz')
        return dict(rows=len(frame))
    def numerical_report():
        r=a.read(OUT/'evaluation/numerical_status.json');a.source(r,'numerical_report.py')
        if r.get('complete') is not True:raise Missing('numerical/cost report incomplete')
        if {(x['end'],x['model']) for x in r.get('coverage',[])}!=set(itertools.product(ENDS,MODELS)):
            raise ValueError('numerical diagnostic generator coverage missing')
        frame=a.table(OUT/'evaluation/numerical_diagnostics.csv')
        if frame.empty or not np.isfinite(frame.value.to_numpy()).all():raise ValueError('empty or invalid numerical diagnostics')
        costs=a.table(OUT/'evaluation/compute_costs.csv');overall=a.table(OUT/'evaluation/overall.csv')
        if len(costs)!=len(overall):raise ValueError('compute cost rows differ from completed cases')
        if not np.isfinite(costs[['training_seconds','replay_seconds','inference_parameters']].to_numpy()).all():raise ValueError('missing measured costs')
        from numerical_report import FIGURES
        figures=a.read(OUT/'evaluation/comparison_figures.json');a.source(figures,'numerical_report.py')
        if figures.get('complete') is not True or set(figures.get('output_sha256',{}))!=set(FIGURES):
            raise Missing('cross-year and pressure figures unfinished')
        for name in FIGURES:a.artifact(OUT/'evaluation'/name)
    a.stage('surface_report',surface_report);a.stage('numerical_report',numerical_report)
    # Always retain the exact required key set, even after missing stages.
    for key in expected_stage_keys():a.coverage.setdefault(key,dict(status='pending',reason='stage not inspected',artifacts={}))
    manifest=audit_coverage(a.coverage,expected_stage_keys(),a.artifacts)
    manifest.update(schema='approved-study-acceptance-v1',expected_keys=list(expected_stage_keys()),
                    producer_sha256=digest(HERE/'acceptance_manifest.py'),accepted_scope='four folds; nine generators; 432 policies; all required audits')
    write_json(OUT/'acceptance_manifest.json',manifest);return manifest


if __name__=='__main__':
    raise SystemExit(0 if build_manifest()['complete'] else 1)
