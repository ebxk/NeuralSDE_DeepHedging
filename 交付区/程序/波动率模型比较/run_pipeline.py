"""Owned producers, provenance-checked recovery and a mandatory final acceptance gate."""
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import threading
from study_core import HERE,OUT,PROJECT,CONFIG,registry,write_json,digest
from artifact_guard import verify_ready,check_hashes,stable_digest
from direct_failure import known_direct_rejection

PY=PROJECT/'工作区/运行支持/环境/volatility/Scripts/python.exe'
TF=PROJECT/'工作区/运行支持/环境/market/python.exe'
OWNED={};LOGS=[]
COMMANDS=set();PROCESS_LOCK=threading.Lock();CANCELLED=threading.Event()

class NumericalRejection(RuntimeError):pass

def process_environment():
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',TF_CPP_MIN_LOG_LEVEL='2')
    for key in ('PYTHONHOME','PYTHONPATH','VIRTUAL_ENV'):env.pop(key,None)
    return env

def command(script,*args,market=False):
    env=process_environment()
    print('STAGE',script,*args,flush=True)
    with PROCESS_LOCK:
        if CANCELLED.is_set():raise RuntimeError('Coordinator cancelled after a producer failure')
        process=subprocess.Popen([str(TF if market else PY),'-u',str(HERE/script),*map(str,args)],env=env,
            stdout=sys.stdout,stderr=sys.stderr,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));COMMANDS.add(process)
    try:
        code=process.wait()
        if code:raise subprocess.CalledProcessError(code,process.args)
    finally:
        with PROCESS_LOCK:COMMANDS.discard(process)

def stop_owned():
    """Stop only this coordinator's live process trees, before another restart."""
    CANCELLED.set()
    with PROCESS_LOCK:processes=list(OWNED.values())+list(COMMANDS)
    for process in processes:
        if process.poll() is not None:continue
        if os.name=='nt':
            subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,timeout=20)
        else:process.terminate()
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:process.kill();process.wait()

def own(role,script,*args,market=False):
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    stdout=(OUT/f'{role}.{stamp}.stdout.log').open('w',encoding='utf8')
    stderr=(OUT/f'{role}.{stamp}.stderr.log').open('w',encoding='utf8');LOGS.extend([stdout,stderr])
    env=process_environment()
    process=subprocess.Popen([str(TF if market else PY),'-u',str(HERE/script),*map(str,args)],
        stdout=stdout,stderr=stderr,env=env,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    OWNED[role]=process
    write_json(OUT/f'{role}.process.json',dict(pid=process.pid,script=script,args=list(args),started=stamp))

def check_producers():
    for role,p in OWNED.items():
        code=p.poll()
        if code is None:continue
        marker=OUT/('surface_stage.json' if role=='surface' else f'queue_seed{role[-1]}.json')
        r=json.loads(marker.read_text(encoding='utf8')) if marker.exists() else {}
        success=r.get('stage_complete') if role=='surface' else r.get('complete')
        if code!=0 or not success:
            write_json(OUT/'producer_failure.json',dict(complete=False,role=role,returncode=code,receipt=r))
            raise RuntimeError(f'{role} producer exited without a successful terminal receipt')

def parallel(function,ends):
    results=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(function,end) for end in ends}
        try:
            while futures:
                done,futures=wait(futures,timeout=15,return_when=FIRST_COMPLETED)
                check_producers()
                for f in done:results.extend(f.result())
        except BaseException:
            for future in futures:future.cancel()
            stop_owned()
            raise
    return results

def archive(path):
    if not path.exists():return
    dest=OUT/'attempts'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')/path.name
    dest.parent.mkdir(parents=True,exist_ok=True)
    if not path.resolve().is_relative_to(OUT.resolve()) or not dest.resolve().is_relative_to(OUT.resolve()):raise ValueError('Unsafe generation archive')
    path.rename(dest)

def ready(end,model):
    folder=OUT/f'inputs/{end}';r=folder/f'sim_{model}.ready.json'
    if not r.exists():return False
    try:verify_ready(r,folder/f'sim_{model}.npz',CONFIG,model,end);return True
    except (ValueError,OSError):return False

def retire_invalid_alias(end,model):
    if not ready(end,model):
        alias=OUT/f'inputs/{end}/sim_{model}.ready.json'
        if alias.exists():archive(alias)

def clean_stage_valid(receipt,output):
    if not receipt.exists() or not output.exists():return False
    try:
        r=json.loads(receipt.read_text(encoding='utf8'))
        if not r.get('complete') or r.get('source')!=digest(HERE/'prepare_clean_base.py'):return False
        check_hashes(r['parents'])
        expected=r['output_sha256'].get(str(output),r['output_sha256'].get(output.name))
        return expected is not None and digest(output)==expected
    except (ValueError,OSError,KeyError):return False

def prepare_inputs():
    from study_core import SHARED_SURFACES,base_factors
    prep=OUT/'data/preparation.json'
    valid=False
    if prep.exists():
        r=json.loads(prep.read_text(encoding='utf8'))
        valid=(r.get('complete') and r.get('source')==digest(HERE/'prepare_data.py')
            and (OUT/'data/quotes.csv').exists() and r.get('quotes_sha256')==digest(OUT/'data/quotes.csv'))
        if valid:check_hashes({str(PROJECT/'工作区/数据/原始数据/50ETF'/k):v for k,v in r['raw_sha256'].items()})
    if not valid:command('prepare_data.py',market=True)
    receipt=OUT/'data/clean_base/status.json'
    if not clean_stage_valid(receipt,SHARED_SURFACES):
        archive(receipt);archive(SHARED_SURFACES)
        command('prepare_clean_base.py','--stage','surfaces')
    for _,_,end in CONFIG['folds']:
        output=base_factors(end);receipt=output.with_suffix('.json')
        if not clean_stage_valid(receipt,output):
            archive(receipt);archive(output)
            command('prepare_clean_base.py','--stage','encoder','--end',end)

def producer_valid(folder,kind,end):
    file=folder/('status.json' if kind in ('heston','sabr') else 'market.json')
    if not file.exists():return False
    try:
        r=json.loads(file.read_text(encoding='utf8'))
        identity=r.get('identity',{})
        if kind in ('heston','sabr'):
            expected=dict(source=digest(HERE/'stochastic_volatility.py'),runner=digest(HERE/'fit_stochastic.py'),
                quotes=digest(OUT/'data/quotes.csv'),fold=list(next(x for x in CONFIG['folds'] if x[-1]==end)))
            if any(identity.get(k)!=v for k,v in expected.items()):return False
        else:
            source='reference_market.py' if kind=='nsde' else 'linear_market.py'
            if identity.get('source')!=digest(HERE/source):return False
            check_hashes(r.get('sources',{}));check_hashes(r.get('parents',identity.get('parents',{})))
        outputs=r.get('output_sha256')
        if not outputs:return False
        check_hashes({str(folder/k):v for k,v in outputs.items()})
        if r.get('status') in ('global_numerical_rejected','numerical_failed'):
            raise NumericalRejection(f'{kind} current-generation numerical rejection: {file}')
        return bool(r.get('complete'))
    except (ValueError,OSError):return False

def blocked(end,model,error,category):
    evidence=next((p for p in (OUT/f'inputs/{end}/sim_{model}.failure.json',
        OUT/f'markets/{end}/{model}/status.json',OUT/f'markets/{end}/{model}/market.json') if p.exists()),None)
    for job in registry():
        if job['end']==end and job['model']==model:
            write_json(OUT/'failures'/f"{job['key']}.json",dict(status='blocked_by_generator',job=job,
                category=category,reason=str(error),producer_sources={str(p):digest(p) for p in HERE.glob('*.py') if not p.name.startswith('test_')}))
            if evidence:
                file=OUT/'failures'/f"{job['key']}.json";r=json.loads(file.read_text(encoding='utf8'))
                source_file=('generator_export.py' if evidence.name.endswith('.failure.json') else
                    'fit_stochastic.py' if model in ('heston','sabr') else
                    'reference_market.py' if model=='nsde' else 'linear_market.py')
                r.update(producer_evidence=dict(path=str(evidence),sha256=digest(evidence),source_file=source_file));write_json(file,r)

def classical_fold(end):
    results=[]
    for model in ('nsde','original_linear','sabr','heston'):
        try:
            retire_invalid_alias(end,model)
            folder=OUT/f'markets/{end}/{model}'
            valid=producer_valid(folder,model,end)
            if not valid:
                archive(folder)
                if model=='nsde':command('reference_market.py','--end',end,market=True)
                elif model=='original_linear':command('linear_market.py','--end',end,market=True)
                else:command('fit_stochastic.py','--end',end,'--kind',model)
            if not producer_valid(folder,model,end):raise RuntimeError(f'Unverified producer {model}/{end}')
            if model=='nsde':command('common_inputs.py','--end',end)
            if not ready(end,model):command('generator_export.py','--end',end,'--model',model)
            if not ready(end,model):raise RuntimeError('Export readiness verification failed')
            results.append(dict(end=end,model=model,status='exported'))
        except Exception as e:
            rejection=OUT/f'markets/{end}/{model}/market.json'
            numeric=rejection.exists() and json.loads(rejection.read_text(encoding='utf8')).get('status')=='numerical_failed'
            category='numerical_rejection' if isinstance(e,NumericalRejection) or numeric else 'infrastructure_failure'
            blocked(end,model,traceback.format_exc(),category)
            if model=='nsde':blocked(end,'gbm',traceback.format_exc(),category)
            results.append(dict(end=end,model=model,status='failed',category=category,reason=traceback.format_exc()))
    return results

def surface_fold(end):
    results=[]
    for model in ('svi_linear','ssvi_linear','essvi_linear','localvol'):
        try:
            retire_invalid_alias(end,model)
            if not ready(end,model):
                if model!='localvol':
                    representation=OUT/f'representations/{end}/{model}.json'
                    valid=False
                    if representation.exists():
                        r=json.loads(representation.read_text(encoding='utf8'))
                        try:
                            check_hashes(r['parents']);valid=r.get('complete') and r.get('source')==digest(HERE/'build_representation.py') and r.get('output_sha256')==digest(representation.with_suffix('.npz'))
                        except (ValueError,OSError):pass
                    if not valid:command('build_representation.py','--end',end,'--kind',model.split('_')[0],market=True)
                    folder=OUT/f'markets/{end}/{model}'
                    if not producer_valid(folder,model,end):
                        archive(folder);command('linear_market.py','--end',end,'--model',model,market=True)
                command('generator_export.py','--end',end,'--model',model)
            if not ready(end,model):raise RuntimeError('Export readiness verification failed')
            results.append(dict(end=end,model=model,status='exported'))
        except Exception as e:
            failures=[OUT/f'inputs/{end}/sim_{model}.failure.json',OUT/f'markets/{end}/{model}/market.json']
            numeric=any(f.exists() and json.loads(f.read_text(encoding='utf8')).get('status')=='numerical_failed' for f in failures)
            category='numerical_rejection' if isinstance(e,NumericalRejection) or numeric else 'infrastructure_failure'
            blocked(end,model,traceback.format_exc(),category)
            results.append(dict(end=end,model=model,status='failed',category=category,reason=traceback.format_exc()))
    return results

def direct_benchmark(end,model):
    status=OUT/f'direct_dynamic/{end}/{model}/status.json'
    def stamp():
        if not status.exists():return None
        stat=status.stat();return stat.st_ino,stat.st_size,stat.st_mtime_ns
    before=stamp()
    try:command('direct_dynamic.py','--end',end,'--model',model)
    except subprocess.CalledProcessError:
        if stamp() is None or stamp()==before:raise
        record=json.loads(status.read_text(encoding='utf8'))
        if not known_direct_rejection(record,end,model):raise
        check_hashes(record['parents']);check_hashes(record['sources'])
        print('RECORDED NUMERICAL REJECTION',end,model,record['error'],flush=True)
        return record

def main():
    CANCELLED.clear()
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'coordinator.lock').open('a+b')
    import msvcrt
    lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    prepare_inputs()
    if (OUT/'generators_finished.json').exists():archive(OUT/'generators_finished.json')
    own('surface','run_surfaces.py','--workers',4,market=True)
    for seed in (0,1,2):own(f'seed{seed}','run_training_queue.py','--seed',seed)
    results=parallel(classical_fold,[x[-1] for x in CONFIG['folds']])
    while OWNED['surface'].poll() is None:
        check_producers();write_json(OUT/'coordinator.json',dict(complete=False,phase='surface_calibration',results=results));time.sleep(15)
    check_producers();results.extend(parallel(surface_fold,[x[-1] for x in CONFIG['folds']]))
    write_json(OUT/'generators_finished.json',dict(complete=True,results=results))
    while any(OWNED[f'seed{s}'].poll() is None for s in (0,1,2)):
        check_producers();write_json(OUT/'coordinator.json',dict(complete=False,phase='policy_training',results=results));time.sleep(15)
    check_producers()
    write_json(OUT/'coordinator.json',dict(complete=False,phase='historical_evaluation',results=results))
    command('verify_accounts.py');command('evaluate_common.py');command('report_common.py')
    write_json(OUT/'coordinator.json',dict(complete=False,phase='audits',results=results))
    for _,_,end in CONFIG['folds']:
        command('direct_hedges.py','--end',end);command('direct_nsde.py','--end',end)
        for model in ('heston','sabr','localvol'):direct_benchmark(end,model)
        for representation in ('original','nsde','svi','ssvi','essvi'):
            if representation in ('original','nsde') or (OUT/f'representations/{end}/{representation}_linear.npz').exists():
                command('factor_quote_audit.py','--end',end,'--representation',representation)
            else:
                write_json(OUT/f'factor_quote_audit/{end}/{representation}/status.json',dict(
                    complete=False,status='failed',error_type='FileNotFoundError',
                    reason='No fitted representation; dependent market failure must be verified by final acceptance'))
        for model in CONFIG['models']:
            if ready(end,model):command('prediction_audit.py','--end',end,'--model',model)
    command('surface_report.py');command('numerical_report.py');command('acceptance_manifest.py');command('finalize_delivery.py')
    command('publish_delivery.py','--push')
    publication=json.loads((OUT/'github_delivery.json').read_text(encoding='utf8'))
    if publication.get('complete') is not True or publication.get('published') is not True:
        raise RuntimeError('GitHub publication has not been verified')
    write_json(OUT/'coordinator.json',dict(complete=True,phase='delivered_and_published',results=results))
    lock.close()

if __name__=='__main__':
    import argparse,sys
    parser=argparse.ArgumentParser();parser.add_argument('--log-file');args=parser.parse_args()
    if args.log_file:
        stream=open(args.log_file,'a',encoding='utf8',buffering=1);sys.stdout=stream;sys.stderr=stream
    try:main()
    except BaseException:
        write_json(OUT/'coordinator_failure.json',dict(complete=False,reason=traceback.format_exc()))
        raise
    finally:
        stop_owned()
        for log in LOGS:log.close()
