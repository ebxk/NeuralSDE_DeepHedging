"""Saved-state static price/IV comparison; no refitting or evaluation tuning."""
import csv
import gzip
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtr
from study_core import OUT,HERE,CONFIG,SHARED_SURFACES,base_factors,bootstrap_legacy,digest,write_json
from volatility_surfaces import price_surface

FACTORS={'original_grid':'original','nsde_factor':'nsde','svi_factor':'svi','ssvi_factor':'ssvi','essvi_factor':'essvi'}
METHODS=('svi_native','ssvi_native','essvi_native','heston','sabr',*FACTORS)
REGIONS=('all','in_domain','short','wings','out_of_tenor')


def independent_iv(t,k,price):
    """Independent 48-step annual-IV bisection, zero r/q, sigma in [0,10]."""
    t,k,p=np.broadcast_arrays(np.asarray(t,float),np.asarray(k,float),np.asarray(price,float))
    def call(sigma):
        root=np.maximum(sigma*np.sqrt(t),1e-30);d=-np.log(k)/root+root/2
        return ndtr(d)-k*ndtr(d-root)
    identifiable=np.isfinite(t)&np.isfinite(k)&np.isfinite(p)&(t>0)&(k>0)&(p>np.maximum(1-k,0)+1e-12)&(p<1)&(p<=call(10.))
    lo=np.zeros(t.shape);hi=np.full(t.shape,10.)
    for _ in range(48):
        mid=(lo+hi)/2;below=call(mid)<p;lo=np.where(below,mid,lo);hi=np.where(below,hi,mid)
    iv=(lo+hi)/2;root=iv*np.sqrt(t);d=-np.log(k)/np.maximum(root,1e-30)+root/2
    vega=np.sqrt(t)*np.exp(-.5*d*d)/np.sqrt(2*np.pi)
    return dict(iv=np.where(identifiable,iv,np.nan),vega=np.where(identifiable,vega,np.nan),identifiable=identifiable)


def region_masks(q):
    t=q.ttm.to_numpy();k=q.k.to_numpy()
    return dict(all=np.ones(len(q),bool),in_domain=(t<=90/365)&(k>=.9)&(k<=1.1),
                short=t<=14/365,wings=(k<.9)|(k>1.1),out_of_tenor=t>90/365)


def score_day(end,split,view,q,predictions):
    """All methods see the same quotes; only masked hidden quotes score heldout."""
    scored=np.ones(len(q),bool) if view=='full' else q.holdout.to_numpy(bool)
    t=q.ttm.to_numpy();k=q.k.to_numpy();target=q.price.to_numpy();market=independent_iv(t,k,target)
    details={};active=[name for name,p in predictions.items() if not p.get('blocked',False)]
    for name,item in predictions.items():
        p=np.full(len(q),np.nan) if item.get('prices') is None else np.asarray(item['prices'],float)
        if p.shape!=(len(q),):raise ValueError('wrong per-quote prediction shape')
        iv=independent_iv(t,k,p);identified=market['identifiable']&iv['identifiable']
        low=identified&((market['vega']<1e-4)|(iv['vega']<1e-4));iv_good=identified&~low
        valid=bool(item.get('valid',False) and np.isfinite(p[scored]).all())
        details[name]=dict(p=p,iv=iv,identified=identified,low=low,iv_good=iv_good,valid=valid,item=item)
    common=bool(active and scored.any() and all(details[name]['valid'] for name in active))
    common_iv=np.logical_and.reduce([details[name]['iv_good'] for name in active]) if active else np.zeros(len(q),bool)
    records=[];quote_rows=[]
    for name,d in details.items():
        item=d['item'];error=d['p']-target;iv_error=d['iv']['iv']-market['iv']
        for region,mask in region_masks(q).items():
            selected=scored&mask;finite=selected&np.isfinite(error);iv_good=selected&d['iv_good']
            matched=selected if common and d['valid'] and not item.get('blocked',False) else np.zeros(len(q),bool)
            matched_iv=matched&common_iv
            records.append(dict(end=int(end),date=str(q.date.iloc[0]),split=split,method=name,view=view,region=region,
                valid=d['valid'],algorithm_valid=bool(item.get('valid',False)),status=item.get('status','unknown'),
                source=item.get('source',''),source_sha256=item.get('source_sha256',''),blocked=bool(item.get('blocked',False)),
                missing=item.get('status') in ('source_error','missing_source'),eligible=bool(selected.any()),
                expected_price_n=int(selected.sum()),price_n=int(finite.sum()),price_sse=float(np.sum(error[finite]**2)),
                iv_n=int(iv_good.sum()),iv_sse=float(np.sum(iv_error[iv_good]**2)),
                low_vega_n=int(np.count_nonzero(selected&d['low'])),unidentifiable_n=int(np.count_nonzero(selected&~d['identified'])),
                matched_date=bool(common and selected.any()),matched_price_n=int(matched.sum()),
                matched_price_sse=float(np.sum(error[matched]**2)),matched_iv_n=int(matched_iv.sum()),
                matched_iv_sse=float(np.sum(iv_error[matched_iv]**2)),matched_methods=';'.join(active)))
        for i in np.flatnonzero(scored):
            quote_rows.append(dict(end=int(end),date=str(q.date.iloc[0]),method=name,view=view,
                code=str(q.code.iloc[i]) if 'code' in q else '',ttm=float(t[i]),k=float(k[i]),holdout=bool(q.holdout.iloc[i]),
                price_observed=float(target[i]),price_predicted=float(d['p'][i]),price_error=float(error[i]),
                iv_observed=float(market['iv'][i]),iv_predicted=float(d['iv']['iv'][i]),iv_error=float(iv_error[i]),
                iv_identifiable=bool(d['identified'][i]),low_vega=bool(d['low'][i]),valid=d['valid'],
                common_qualified=common,common_iv_eligible=bool(common_iv[i] and common),status=item.get('status','unknown')))
    return records,quote_rows


class Aggregator:
    def __init__(self):self.rows={}
    def add(self,records):
        for r in records:
            key=tuple(r[k] for k in ('end','split','method','view','region'))
            if key not in self.rows:
                self.rows[key]=dict(zip(('end','split','method','view','region'),key),days=0,eligible_days=0,valid_days=0,
                    failed_days=0,missing_days=0,blocked_days=0,expected_price_n=0,price_n=0,price_sse=0.,iv_n=0,iv_sse=0.,
                    all_attempt_price_n=0,all_attempt_price_sse=0.,all_attempt_iv_n=0,all_attempt_iv_sse=0.,
                    low_vega_n=0,unidentifiable_n=0,matched_dates=0,matched_price_n=0,matched_price_sse=0.,matched_iv_n=0,matched_iv_sse=0.,matched_methods=r['matched_methods'])
            x=self.rows[key];x['days']+=1;x['eligible_days']+=int(r['eligible'])
            for field in ('expected_price_n','low_vega_n','unidentifiable_n'):x[field]+=r[field]
            for field in ('price_n','price_sse','iv_n','iv_sse'):x['all_attempt_'+field]+=r[field]
            if r['valid']:
                x['valid_days']+=int(r['eligible'])
                for field in ('price_n','price_sse','iv_n','iv_sse'):x[field]+=r[field]
            elif r['eligible']:
                x['blocked_days']+=int(r['blocked']);x['missing_days']+=int(r['missing'])
                x['failed_days']+=int(not r['blocked'] and not r['missing'])
            x['matched_dates']+=int(r['matched_date'])
            for field in ('matched_price_n','matched_price_sse','matched_iv_n','matched_iv_sse'):x[field]+=r[field]
    def frame(self):
        rows=[]
        for record in self.rows.values():
            r=dict(record)
            for prefix in ('','all_attempt_','matched_'):
                for metric in ('price','iv'):
                    n=r[prefix+metric+'_n'];r[prefix+metric+'_rmse']=float(np.sqrt(r[prefix+metric+'_sse']/n)) if n else None
            r['coverage']=r['valid_days']/r['eligible_days'] if r['eligible_days'] else None
            rows.append(r)
        return pd.DataFrame(rows)


def aggregate_daily(records):
    a=Aggregator();a.add(records);return a.frame()


class Reader:
    def __init__(self):
        from acceptance_manifest import Auditor
        self.audit=Auditor()
    def read(self,path):return self.audit.read(path)
    def asset(self,path):return self.audit.artifact(path)


def native_prediction(reader,kind,q,view):
    path=OUT/f'surfaces/{kind}/{q.date.iloc[0]}/{view}.json';r=reader.read(path)
    reader.audit.source(r,'volatility_surfaces.py')
    identity=r.get('identity',{})
    if identity.get('view')!=view or identity.get('kind')!=kind or identity.get('day')!=q.date.iloc[0]:
        raise ValueError('native full/masked date/view provenance mismatch')
    if identity.get('quotes'):reader.audit.artifact(OUT/'data/quotes.csv',identity['quotes'])
    if identity.get('runner'):reader.audit.artifact(HERE/'run_surfaces.py',identity['runner'])
    if r.get('status') not in ('complete','numerical_rejected') or 'model' not in r:raise ValueError('native source lacks reconstructable saved parameters')
    p=price_surface(r['model'],q.ttm.to_numpy(),q.k.to_numpy())
    return dict(prices=p,valid=bool(r['model']['diagnostics']['valid']),status=r['status'],source=str(path),source_sha256=reader.asset(path))


class Provider:
    def __init__(self,reader,end,method):
        self.reader=reader;self.end=end;self.method=method;self.blocked=False;self.failure=None
        if method.endswith('_native'):return
        try:
            if method in ('heston','sabr'):
                directory=OUT/f'markets/{end}/{method}';r=reader.read(directory/'global.json')
                reader.audit.source(r,'stochastic_volatility.py');self.model=r['model']
                status=reader.read(directory/'status.json')
                if status.get('complete') is not True or not self.model['diagnostics']['valid']:raise ValueError('global Q model rejected')
                self.state_source=directory/'states.csv';self.state_sha=reader.asset(self.state_source)
                self.states=pd.read_csv(self.state_source).set_index('date')
                if self.states.index.duplicated().any():raise ValueError('duplicate stochastic date state')
            else:
                representation=FACTORS[method];directory=OUT/f'factor_quote_audit/{end}/{representation}'
                receipt=reader.read(directory/'status.json');reader.audit.source(receipt,'factor_quote_audit.py')
                if receipt.get('status') not in ('complete','fit_failed'):raise ValueError('factor audit not executed')
                reader.asset(directory/'daily.csv');self.flags=pd.read_csv(directory/'daily.csv').set_index(['date','view'])
                if self.flags.index.duplicated().any():raise ValueError('duplicate factor view/date')
                self.state_source=directory/'daily.jsonl';self.state_sha=reader.asset(self.state_source);self.saved={}
                with (directory/'daily.jsonl').open(encoding='utf8') as handle:
                    for line in handle:
                        record=json.loads(line);key=(record['date'],record['view'])
                        if key in self.saved:raise ValueError('duplicate saved factor state')
                        self.saved[key]={k:record[k] for k in ('xi','grid','valid') if k in record}
                if representation=='original':path=SHARED_SURFACES;keys=('grid_t','grid_k')
                else:
                    path=base_factors(end) if representation=='nsde' else OUT/f'representations/{end}/{representation}_linear.npz'
                    keys=('grid_t','grid_k','center','basis')
                reader.asset(path)
                with np.load(path,allow_pickle=False) as data:self.rep={k:data[k] for k in keys}
        except Exception as error:
            self.failure=f'{type(error).__name__}: {error}'
            generator=method if method in ('heston','sabr') else FACTORS.get(method,'')+'_linear'
            missing_basis=method in ('svi_factor','ssvi_factor','essvi_factor') and not (OUT/f'representations/{end}/{FACTORS[method]}_linear.npz').exists()
            if generator in ('heston','sabr') or missing_basis:
                try:
                    declared=reader.audit.blocked(end,generator)
                    self.blocked=declared.get('status')=='declared_failure'
                except Exception:pass # Original error remains explicit; no false scientific closure.
    def predict(self,q,view):
        if self.failure:
            return dict(prices=None,valid=False,status='blocked' if self.blocked else 'missing_source',blocked=self.blocked,error=self.failure)
        if self.method.endswith('_native'):return native_prediction(self.reader,self.method.split('_')[0],q,view)
        if self.method in ('heston','sabr'):
            from stochastic_volatility import price_model
            row=self.states.loc[q.date.iloc[0]];state=float(row['state' if view=='full' else 'masked_state'])
            return dict(prices=price_model(self.method,self.model,q.ttm.to_numpy(),q.k.to_numpy(),state),valid=True,status='complete',
                        source=str(self.state_source),source_sha256=self.state_sha)
        bootstrap_legacy();from extended_curve import curve
        key=(q.date.iloc[0],view);saved=self.saved[key];flag=self.flags.loc[key]
        if bool(saved['valid'])!=bool(flag['valid']):raise ValueError('factor saved-state validity disagrees with daily table')
        rep=dict(self.rep)
        if self.method=='original_grid':rep.update(center=np.array(saved['grid']),basis=np.zeros((2,len(saved['grid']))));xi=np.zeros(2)
        else:xi=np.array(saved['xi'])
        prices=curve(q.ttm.to_numpy(),q.k.to_numpy(),np.broadcast_to(xi,(len(q),2)),rep,np.ones(2))[0]
        return dict(prices=prices,valid=bool(saved['valid']),status='complete' if saved['valid'] else 'fit_failed',
                    source=str(self.state_source),source_sha256=self.state_sha)


def _split(day,fold):
    return 'training' if day<str(fold[0]+1) else 'validation' if day<str(fold[1]+1) else 'historical_evaluation'


def build_report():
    started=time.perf_counter();directory=OUT/'evaluation';directory.mkdir(parents=True,exist_ok=True)
    status_path=directory/'surface_status.json';reader=Reader();errors=[];missing=[];aggregate=Aggregator()
    write_json(status_path,dict(complete=False,status='running'))
    try:
        reader.asset(OUT/'data/quotes.csv');quotes=pd.read_csv(OUT/'data/quotes.csv').sort_values(['date','ttm','k','code']).reset_index(drop=True)
        if quotes.duplicated(['date','expire_date','strike']).any():raise ValueError('duplicate effective quote in shared target')
        daily_tmp=directory/'surface_daily.csv.tmp';quote_tmp=directory/'quote_errors.csv.gz.tmp';daily_writer=quote_writer=None
        with daily_tmp.open('w',encoding='utf8',newline='') as dayfile,gzip.open(quote_tmp,'wt',encoding='utf8',newline='') as quotefile:
            for fold in CONFIG['folds']:
                end=fold[-1];providers={method:Provider(reader,end,method) for method in METHODS}
                for method,p in providers.items():
                    if p.failure:
                        missing.append(dict(end=end,method=method,blocked=p.blocked,reason=p.failure))
                        if not p.blocked:errors.append(dict(end=end,method=method,error=p.failure))
                for day,q in quotes[quotes.date<str(end+1)].groupby('date',sort=True):
                    for view in ('full','heldout'):
                        predictions={}
                        for method,p in providers.items():
                            try:predictions[method]=p.predict(q,view)
                            except Exception as error:
                                message=f'{type(error).__name__}: {error}'
                                errors.append(dict(end=end,method=method,date=day,view=view,error=message))
                                predictions[method]=dict(prices=None,valid=False,status='source_error')
                        daily,per_quote=score_day(end,_split(day,fold),view,q,predictions);aggregate.add(daily)
                        if daily_writer is None:
                            daily_writer=csv.DictWriter(dayfile,fieldnames=list(daily[0]));daily_writer.writeheader()
                            quote_writer=csv.DictWriter(quotefile,fieldnames=list(per_quote[0]));quote_writer.writeheader()
                        daily_writer.writerows(daily);quote_writer.writerows(per_quote)
        daily_tmp.replace(directory/'surface_daily.csv');quote_tmp.replace(directory/'quote_errors.csv.gz')
        comparison=aggregate.frame();tmp=directory/'surface_comparison.csv.tmp';comparison.to_csv(tmp,index=False);tmp.replace(directory/'surface_comparison.csv')
        if comparison.empty:errors.append(dict(error='no reconstructed static quote results'))
        known={(row['end'],row['method']) for row in missing}
        for error in errors:
            identity=(error.get('end'),error.get('method'))
            if identity[0] is not None and identity[1] is not None and identity not in known:
                missing.append(dict(end=identity[0],method=identity[1],blocked=False,reason=error['error']));known.add(identity)
        from artifact_guard import check_hashes
        check_hashes(reader.audit.artifacts)
        result=dict(complete=not errors,status='complete' if not errors else 'failed_source_coverage',
            scope='reconstructed frozen-state normalized price and independent IV; five overlapping domains',
            methods=list(METHODS),missingmethods=missing,source_errors=errors,
            parents=reader.audit.artifacts,source=digest(HERE/'surface_report.py'),
            output_sha256={name:digest(directory/name) for name in ('surface_daily.csv','surface_comparison.csv','quote_errors.csv.gz')},
            comparison_rows=len(comparison),seconds=time.perf_counter()-started,
            matched_rule='all nonblocked methods qualified on same dates; IV additionally same identifiable high-Vega quote intersection',
            iv_rule='independent 48-step BS bisection [0,10]; exclude joint Vega<1e-4 or unidentifiable; counts explicit',
            views='full all quotes descriptive; heldout only hidden quotes from independent masked states',
            regions=dict(in_domain='T<=90/365 and .9<=k<=1.1',short='T<=14/365',wings='k<.9 or k>1.1',out_of_tenor='T>90/365'),
            capacity_note='native SVI families and two-factor families have different capacity; NN drift does not alter static price family',
            domain_note='original/factor 14/30/60/90-day grids retain legacy extrapolation; native maturity slices differ; inspect domain groups separately',
            domains_overlap=True,git_allowed=['surface_comparison.csv'],local_only=['surface_daily.csv','quote_errors.csv.gz'])
        write_json(status_path,result);return result
    except Exception as error:
        errors.append(dict(error=f'{type(error).__name__}: {error}'))
        result=dict(complete=False,status='failed_source_coverage',source_errors=errors,missingmethods=missing,
                    parents=reader.audit.artifacts,source=digest(HERE/'surface_report.py'),seconds=time.perf_counter()-started)
        write_json(status_path,result);return result


if __name__=='__main__':raise SystemExit(0 if build_report()['complete'] else 1)
