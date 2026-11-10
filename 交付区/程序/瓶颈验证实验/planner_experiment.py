"""Budgeted terminal-rollout diagnostics and closed-loop execution stresses.

The same-information control has 35 inputs: base8, returnFIFO20, raw residual,
current BS delta/validity, current spot-unit/original notional, causally earned
pending dividend rights, own wealth and actual position. Two hidden layers46.
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

FROZEN={}
BRIDGE=None


def read_json(path):
    # Brief sharing denials are observed on both sides of the Windows file queue.
    deadline=time.monotonic()+5
    while True:
        try:return json.loads(Path(path).read_text())
        except PermissionError:
            if time.monotonic()>=deadline:raise
            time.sleep(.05)


class BudgetExceeded(RuntimeError):pass


class Budget:
    def __init__(self,out):
        self.path=out/'planner/budget.json';self.path.parent.mkdir(exist_ok=True,parents=True)
        self.used=json.loads(self.path.read_text()).get('seconds',0) if self.path.exists() else 0
        self.started=time.perf_counter()
    def checkpoint(self):
        now=time.perf_counter();self.used+=now-self.started;self.started=now
        write(self.path,dict(seconds=self.used,limit_seconds=86400,maximum_workers=1,
            status='budget_incomplete' if self.used>=86400 else 'running'))
        if self.used>=86400:raise BudgetExceeded('Registered planning queue exhausted 24 active hours')


def write(path,data):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf8');temp.replace(path)


def accepted_action(selected,baseline):
    difference=np.square(selected)-np.square(baseline)
    mean=float(difference.mean());upper=mean+1.6448536269514722*float(difference.std(ddof=1))/np.sqrt(len(difference))
    relative=-mean/max(float(np.mean(np.square(baseline))),1e-20)
    return bool(upper<0 and relative>=.01),dict(mean_mse_difference=mean,upper_one_sided95=upper,
            relative_mse_improvement=relative,confirmation_paths=len(difference))


def continuation_pnl(model,data,state,first_action,fee,delay=False):
    import torch
    from extended_policy import _at
    s=data['spot'];n,h=data['active'].shape
    cash=torch.full((n,),state['cash'],dtype=s.dtype)
    shares=torch.full((n,),state['shares'],dtype=s.dtype)
    unit=state['unit'];notional=state['notional'];receivable=state['receivable']
    if hasattr(model,'execution_delay'):model.execution_delay=delay
    cost=torch.zeros_like(cash);prepared=model.prepare(data)
    pending=first_action.to(s.dtype)*unit
    for t in range(h):
        wealth=(cash+receivable+shares*s[:,t])/notional
        if delay:
            target=shares if t==0 else pending
        else:
            ratio=first_action if t==0 else model.action(_at(prepared,t),shares.to(torch.float32)/unit,wealth)[0]
            target=ratio.to(s.dtype)*unit
        traded=target-shares
        charge=fee*traded.abs()*s[:,t];cost+=charge/notional
        cash=cash-traded*s[:,t]-charge;shares=target
        if delay and t>0:
            post=(cash+receivable+shares*s[:,t])/notional
            pending=model.action(_at(prepared,t),shares.to(torch.float32)/unit,post)[0].to(s.dtype)*unit
    charge=fee*shares.abs()*s[:,-1];cost+=charge/notional
    cash+=shares*s[:,-1]-charge-unit*torch.relu(s[:,-1]-state['strike'])
    return (cash+receivable+state.get('pending_asset',0))/notional,cost


def setup(project):
    program=project/'工作区/当前研究/程序'
    sys.path[:0]=[str(program/'瓶颈验证实验'),str(program/'扩展组合实验'),
                  str(program/'创新组合实验'),str(program/'完整认购')]
    return program


def child_environment():
    # uv's 3.13 launcher exports its stdlib home; never leak it into TF's 3.9.
    environment={k:v for k,v in os.environ.items() if k not in ('PYTHONHOME','UV_INTERNAL__PYTHONHOME')}
    environment.update(PYTHONUTF8='1',TF_CPP_MIN_LOG_LEVEL='2')
    return environment


def raw_real(paths,common,reference,market):
    ix={d:i for i,d in enumerate(reference['real_dates'])};output=[]
    for path in paths:
        indices=np.array([ix[row['date']] for row in path]);x=common['real_xi'][indices]
        dt=np.diff(np.array([r['date'] for r in path],dtype='datetime64[D]')).astype(float)/365
        residual=np.diff(x,axis=0)-reference['mu'][indices[:-1],1:]*dt[:,None]
        covariance=reference['cov'][indices[:-1],1:,1:]*dt[:,None,None]+1e-6*np.eye(2)
        errors=np.sum(residual*np.linalg.solve(covariance,residual[...,None])[...,0],axis=1)
        q=np.empty(len(path));q[0]=market['initial']
        for t in range(1,len(q)):q[t]=.94*q[t-1]+.06*errors[t-1]
        output.append(q)
    return output


def augment_real(paths,data,inputs,out,end):
    import torch
    from extended_curve import bs_delta
    reference=dict(np.load(out/'markets'/str(end)/'reference_coefficients.npz'))
    market=json.loads((out/'markets'/str(end)/'reference_market.json').read_text())
    raw=raw_real(paths,inputs.common,reference,market)
    n,h=data['active'].shape
    q=np.empty((n,h),dtype='f');fifo=np.empty((n,h,20),dtype='f')
    calendar={d:i for i,d in enumerate(inputs.common['return_dates'])}
    for i,p in enumerate(paths):
        q[i,:len(p)-1]=raw[i][:-1];q[i,len(p)-1:]=raw[i][-1]
        for t in range(h):
            date=p[min(t,len(p)-1)]['date'];idx=calendar[date]
            fifo[i,t]=inputs.common['returns'][idx-19:idx+1]
    strike=np.empty((n,h));quote=np.empty((n,h));tau=np.zeros((n,h))
    for i,p in enumerate(paths):
        for t in range(h):
            row=p[min(t,len(p)-1)]
            strike[i,t]=float(row['strike']);quote[i,t]=float(row['option_close'])
            tau[i,t]=max(0,(np.datetime64(p[-1]['date'])-np.datetime64(row['date'])).astype(float)/365)
    delta,valid=bs_delta(data['spot'][:,:-1],strike,quote,tau)
    return dict(data,raw_ewma=torch.from_numpy(q),return_fifo=torch.from_numpy(fifo),
        delta=torch.where(valid,delta,0),delta_valid=valid,
        notional_scale=(data['spot']*data['unit']/(data['spot'][:,0:1]*data['unit'][:,0:1]))[:,:-1])


def state_snapshot(model,data,paths,inputs,index,cut,out,end,fee=.0005):
    import torch
    from continuation_account import account_rollout
    one={k:v[index:index+1] for k,v in data.items()}
    _,_,state=account_rollout(model,one,fee,stop=cut)
    entitlements=state['entitlements']; cash=float(state['cash'][0]);recv=float(state['receivable'][0])
    pending=0.
    for event,amount in entitlements.items():
        if one['ex_event'][0,cut].item()==event:recv+=float(amount[0])
        if one['pay_event'][0,cut].item()==event:
            recv-=float(amount[0]);cash+=float(amount[0])
        if not (one['ex_event'][0,:cut+1]==event).any():pending+=float(amount[0])
    p=paths[index];row=p[cut]
    xi_index=np.searchsorted(inputs.common['real_dates'],row['date'])
    return dict(date=row['date'],episode=row['episode'],spot=float(row['spot']),strike=float(row['strike']),
        quote=float(row['option_close']),unit=float(row['unit']),cash=cash,receivable=recv,pending_asset=pending,
        shares=float(state['shares'][0]),notional=float(state['notional'][0]),past_cost=float(state['cost'][0]),
        raw_ewma=float(one['raw_ewma'][0,cut]),warm=one['return_fifo'][0,cut].tolist(),
        delta=float(one['delta'][0,cut]),delta_valid=bool(one['delta_valid'][0,cut]),
        xi=inputs.common['real_xi'][xi_index].tolist(),f=one['f'][0,cut].tolist(),
        dates=[r['date'] for r in p[cut:]],cut=cut,path_index=index)


def select_states(paths,data,training=False):
    candidates=[]
    dates=sorted({r['date'] for p in paths for r in p[:-1]})
    threshold=dates[int(.9*len(dates))] if training else dates[0]
    for i,p in enumerate(paths):
        for t,r in enumerate(p[:-1]):
            if r['date']<threshold:continue
            f=data['f'][i,t].numpy();tau=float(f[1]/12)
            group=(min(int(tau*365//30),3),min(int(float(f[6])//1),3))
            candidates.append((group,i,t))
    rng=np.random.default_rng(20261112)
    rng.shuffle(candidates);groups={}
    for group,i,t in candidates:groups.setdefault(group,[]).append((i,t))
    chosen=[]
    while len(chosen)<32 and any(groups.values()):
        for group in sorted(groups):
            if groups[group] and len(chosen)<32:chosen.append(groups[group].pop())
    if len(chosen)!=32:raise ValueError('Insufficient registered diagnostic states')
    return chosen


def generate(project,out,end,state_path,count,phase):
    """TF-only subprocess; never imports torch or future quote/term arrays."""
    setup(project)
    from conditional_market import load_frozen
    from market import simulate,reliability
    key=(str(project),end)
    if key not in FROZEN:FROZEN[key]=load_frozen(project,end)
    fit,geom,rep,common,cfg,cal,base,identity=FROZEN[key]
    state=read_json(state_path)
    t=(np.array(state['dates'],dtype='datetime64[D]')-np.datetime64(state['date'])).astype(float)/365
    seed=20261113+phase
    noise=np.random.default_rng(seed).normal(size=(count,len(t)-1,2,3))
    start=np.tile([np.log(state['spot']),*state['xi']],(count,1))
    times=np.tile(t,(count,1))
    paths,diagnostic=simulate(fit,start,times,noise,geom,2,'reflect')
    raw=reliability(paths[:,:,1:],times,fit,geom,state['raw_ewma'])
    raw[:,0,2]=state['f'][7]
    market=json.loads((base/'C1/market.json').read_text())
    from conditional_market import curve_price
    spot=np.exp(paths[:,:,0]);tau=times[:,-1:]-times
    prices=curve_price(spot,state['strike'],tau,paths[:,:,1:],rep,common['scale'],project)
    delta,valid=delta_numpy(spot,state['strike'],prices,tau)
    delta[:,0]=state['delta'];valid[:,0]=state['delta_valid']
    array_path=state_path.with_suffix(f'.{count}.npz')
    temporary=array_path.with_suffix('.tmp.npz')
    np.savez(temporary,spot=spot,xi=paths[:,:,1:],raw=raw,time=times,delta=delta,delta_valid=valid,
             lower=market['lower'],upper=market['upper'],center=market['center'],spread=market['spread'])
    temporary.replace(array_path)
    from prepare_common import digest
    write(state_path.with_suffix(f'.{count}.json'),dict(count=count,seed=seed,source=identity,diagnostics=diagnostic,
        state_sha256=digest(state_path),program_sha256=digest(Path(__file__)),array_sha256=digest(array_path)))


def delta_numpy(spot,strike,quote,tau):
    from scipy.special import ndtr
    valid=(tau>0)&(quote>np.maximum(spot-strike,0))&(quote<spot)&np.isfinite(quote)
    result=np.zeros_like(spot)
    lo=np.full(valid.sum(),1e-8);hi=np.full(valid.sum(),12.)
    money=np.log(spot[valid]/strike);q=quote[valid]/spot[valid];k=strike/spot[valid]
    for _ in range(40):
        v=(lo+hi)/2;d=money/v+v/2;c=ndtr(d)-k*ndtr(d-v)
        lo=np.where(c<q,v,lo);hi=np.where(c<q,hi,v)
    v=(lo+hi)/2;result[valid]=ndtr(money/v+v/2)
    return result.astype('f'),valid


def serve(project,out,end,queue):
    queue.mkdir(exist_ok=True,parents=True)
    while not (queue/'stop').exists():
        for request in queue.glob('*.request.json'):
            task=read_json(request);done=request.with_suffix('.done.json')
            try:
                generate(project,out,end,Path(task['state']),task['count'],task['phase'])
                write(done,dict(status='complete'))
            except Exception as e:
                write(done,dict(status='failed',reason=repr(e)))
            request.rename(request.with_suffix('.processed.json'))
        time.sleep(.05)


class MarketBridge:
    def __init__(self,project,out,end,budget):
        self.queue=out/'planner'/str(end)/f'queue_{os.getpid()}_{time.time_ns()}'
        self.queue.mkdir(exist_ok=True,parents=True)
        self.budget=budget;self.number=0
        self.log=(self.queue/'worker.log').open('wb')
        self.process=subprocess.Popen([str(project/'工作区/运行支持/环境/market/python.exe'),'-u',str(Path(__file__).resolve()),
            '--project',str(project),'--out',str(out),'--end',str(end),'--stage','serve','--queue',str(self.queue)],
            env=child_environment(),stdout=self.log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    def request(self,path,count,phase):
        request=self.queue/f'{self.number}.request.json';self.number+=1
        write(request,dict(state=str(path),count=count,phase=phase))
        done=request.with_suffix('.done.json')
        while True:
            self.budget.checkpoint()
            if done.exists():
                try:
                    record=read_json(done)
                    break
                except PermissionError:
                    # Windows may briefly deny a just-renamed cross-process file.
                    time.sleep(.05)
                    continue
            if self.process.poll() is not None:raise RuntimeError('TF planner worker exited; see worker.log')
            time.sleep(.1)
        if record['status']!='complete':raise RuntimeError(record['reason'])
    def close(self):
        (self.queue/'stop').touch()
        try:self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:self.process.terminate();self.process.wait(timeout=10)
        self.log.close()


def export_raw(project,out,end):
    setup(project)
    from conditional_market import load_frozen
    from market import reliability
    fit,geom,rep,common,cfg,cal,base,identity=load_frozen(project,end)
    market=json.loads((base/'C1/market.json').read_text())
    with np.load(base/'C1/single.npz') as sim:
        raw=reliability(sim['xi'].astype(float),common['time'].astype(float),fit,geom,market['initial'])[:,:,0]
    path=out/'planner'/str(end)/'raw_sim_quality.npz'
    path.parent.mkdir(exist_ok=True,parents=True);np.savez(path,raw=raw)
    from prepare_common import digest
    write(path.with_suffix('.json'),dict(identity=identity,shape=list(raw.shape),program_sha256=digest(Path(__file__)),
        source_single_sha256=digest(base/'C1/single.npz'),array_sha256=digest(path)))


class FullInputs:
    def __init__(self,inputs,out,end):
        self.inputs=inputs;self.out=out;self.end=end;self.folds=inputs.folds
        self.sim=None;self.real={}
    def simulation(self,*args):
        import torch
        if self.sim is None:
            data,ids=self.inputs.simulation(args[0] if args else 'single',True)
            warm=self.inputs.common['warm'];spot=data['spot'].numpy()
            ret=np.concatenate((warm,np.diff(np.log(spot),axis=1)),axis=1)
            fifo=np.lib.stride_tricks.sliding_window_view(ret,20,axis=1)[:,:-1].copy()
            raw=np.load(self.out/'planner'/str(self.end)/'raw_sim_quality.npz')['raw'][:,:-1].copy()
            self.sim=(dict(data,return_fifo=torch.from_numpy(fifo.astype('f')),
                           raw_ewma=torch.from_numpy(raw),delta=torch.where(data['delta_valid'],data['delta'],0),
                           notional_scale=(data['spot']/data['spot'][:,0:1])[:,:-1]),ids)
        return self.sim
    def observed(self,year,*args,**kwargs):
        if year not in self.real:
            paths,data=self.inputs.observed(year,*args,**kwargs)
            self.real[year]=(paths,augment_real(paths,data,self.inputs,self.out,self.end))
        return self.real[year]


def train_fullstate(project,out,end,budget,only_seed=None):
    import torch
    from fullstate_policy import FullStatePolicy
    from bottleneck_train import inputs_for,job_for,evaluate
    from extended_train import fit
    from prepare_common import digest
    rawfile=out/'planner'/str(end)/'raw_sim_quality.npz'
    raw_ok=False
    if rawfile.exists() and rawfile.with_suffix('.json').exists():
        meta=json.loads(rawfile.with_suffix('.json').read_text())
        raw_ok=(meta.get('program_sha256')==digest(Path(__file__)) and meta.get('array_sha256')==digest(rawfile))
    if not raw_ok:
        subprocess.run([str(project/'工作区/运行支持/环境/market/python.exe'),'-u',str(Path(__file__).resolve()),
            '--project',str(project),'--out',str(out),'--end',str(end),'--stage','raw'],check=True,
            timeout=max(1,86400-budget.used),env=child_environment())
    cfg=json.loads((Path(__file__).parent/'study.json').read_text())
    inputs=FullInputs(inputs_for(project,dict(end=end,kind='plain'),out),out,end)
    results=[]
    for seed in (range(3) if only_seed is None else (only_seed,)):
        if only_seed is None:
            budget.checkpoint()
            subprocess.run([str(project/'工作区/运行支持/环境/hedge313/Scripts/python.exe'),'-u',str(Path(__file__).resolve()),
                '--project',str(project),'--out',str(out),'--end',str(end),'--stage','fit','--seed',str(seed)],
                check=True,timeout=max(1,86400-budget.used),env=child_environment())
            budget.checkpoint()
            status=out/'policies'/f'{end}_fullstate_s{seed}.json'
            record=json.loads(status.read_text())
            results.append(dict(key=f'{end}_fullstate_s{seed}',weights=str(status.with_suffix('.pt')),
                                validation=record['validation']))
            continue
        budget.checkpoint();torch.manual_seed(seed);torch.set_num_threads(1)
        task=dict(end=end,kind='fullstate',seed=seed,key=f'{end}_fullstate_s{seed}')
        job=job_for(task);model=FullStatePolicy()
        initial=out/'initial'/f'{end}_fullstate_s{seed}.pt'
        if not initial.exists():torch.save(model.state_dict(),initial)
        model.load_state_dict(torch.load(initial,weights_only=True))
        path=out/'policies'/f'{task["key"]}.pt';status=path.with_suffix('.json')
        identity=dict(task=task,initial=digest(initial),config=cfg,source={p.name:digest(p) for p in
            [Path(__file__),Path(__file__).with_name('fullstate_policy.py')]},
            inputs={str(p):digest(p) for p in [inputs.inputs.episode_file,
                inputs.inputs.directory/'single.npz',inputs.inputs.directory/'common.npz',rawfile]})
        if status.exists():
            record=json.loads(status.read_text())
            if record['identity']!=identity or not path.exists() or digest(path)!=record['weights_sha256']:
                raise ValueError('Stale fullstate artifact preserved; require explicit new version')
        else:
            state,stats=fit(model,job,cfg,inputs)
            temp=path.with_suffix('.tmp');torch.save(state,temp);temp.replace(path)
            model.load_state_dict(torch.load(path,weights_only=True))
            _,val=inputs.observed(inputs.folds[1]);metrics=evaluate(model,job,val)
            if abs(metrics['mse']*1e4-stats['best'])>1e-6:raise AssertionError('Fullstate reload changed risk')
            record=dict(status='complete',identity=identity,weights_sha256=digest(path),validation=metrics,**stats)
            write(status,record)
        results.append(dict(key=task['key'],weights=str(path),validation=record['validation']))
    return dict(status='complete',registered_training_tasks=results)


def conditional_data(state,arrays):
    import torch
    from hedge_experiment import features
    s=arrays['spot'];returns=np.diff(np.log(s),axis=1)
    past=np.concatenate((np.tile(state['warm'],(len(s),1)),returns),axis=1)
    windows=np.lib.stride_tricks.sliding_window_view(past,20,axis=1)
    rv=np.sqrt(252*np.mean(windows**2,axis=-1))
    q=arrays['raw'].copy();q[...,:2]=(np.clip(q[...,:2],arrays['lower'],arrays['upper'])-arrays['center'])/arrays['spread']
    tau=arrays['time'][:,-1:]-arrays['time']
    f=features(s,state['strike'],tau,rv,np.concatenate((arrays['xi'],q),axis=-1))[:,:-1]
    # Initial inputs must equal the genuine real prefix, not a new contract start.
    np.testing.assert_allclose(f[:,0],np.tile(state['f'],(len(s),1)),rtol=2e-5,atol=2e-5)
    return dict(f=torch.from_numpy(f),spot=torch.from_numpy(s),unit=torch.full_like(torch.from_numpy(s),state['unit']),
                active=torch.ones((len(s),s.shape[1]-1),dtype=torch.bool),
                return_fifo=torch.from_numpy(windows[:,:-1].copy().astype('f')),
                raw_ewma=torch.from_numpy(arrays['raw'][:,:-1,0].astype('f')),
                delta=torch.from_numpy(arrays['delta'][:,:-1]),delta_valid=torch.from_numpy(arrays['delta_valid'][:,:-1]),
                notional_scale=torch.from_numpy(state['unit']*s[:,:-1]/state['notional']),
                initial_notional=torch.full((len(s),),state['notional'],dtype=torch.float64),
                initial_pending_asset=torch.full((len(s),),state['pending_asset'],dtype=torch.float64))


def model_for(out,end,seed,fullstate=False):
    import torch
    if fullstate:
        from fullstate_policy import FullStatePolicy
        model=FullStatePolicy();key=f'{end}_fullstate_s{seed}'
    else:
        from extended_policy import make_policy
        model=make_policy('wide');key=f'{end}_plain_s{seed}'
    model.load_state_dict(torch.load(out/'policies'/f'{key}.pt',weights_only=True))
    model.eval();return model


def check_state(project,out,end,seed,state,model,path,budget,delay=False):
    import torch
    from extended_policy import _at
    from extended_curve import bs_delta
    from prepare_common import digest
    for count,phase in ((256,0),(1024,1)):
        npz=path.with_suffix(f'.{count}.npz')
        if not npz.exists() or not path.with_suffix(f'.{count}.json').exists():
            budget.checkpoint()
            if BRIDGE is not None:BRIDGE.request(path,count,phase)
            else:
                subprocess.run([str(project/'工作区/运行支持/环境/market/python.exe'),'-u',str(Path(__file__).resolve()),
                    '--project',str(project),'--out',str(out),'--end',str(end),'--stage','generate',
                    '--state',str(path),'--count',str(count),'--phase',str(phase)],check=True,
                    timeout=max(1,86400-budget.used),env=child_environment())
            budget.checkpoint()
        meta=read_json(path.with_suffix(f'.{count}.json'))
        if (meta.get('program_sha256')!=digest(Path(__file__)) or meta.get('state_sha256')!=digest(path)
            or meta.get('array_sha256')!=digest(npz)):
            raise ValueError('Conditional path cache does not match frozen state/source')
        data=conditional_data(state,dict(np.load(npz)))
        prepared=model.prepare(data);dtype=next(model.parameters()).dtype
        previous=state['shares']/state['unit']
        wealth=(state['cash']+state['receivable']+state['shares']*state['spot'])/state['notional']
        base=float(model.action(_at(prepared,0)[:1],torch.tensor([previous],dtype=dtype),torch.tensor([wealth]))[0][0])
        if count==256:
            d,valid=bs_delta(state['spot'],state['strike'],state['quote'],(np.datetime64(state['dates'][-1])-np.datetime64(state['date'])).astype(float)/365)
            actions=[base,previous,base-.05,base+.05,base-.15,base+.15]
            if bool(valid):actions.append(float(d))
            actions=list(dict.fromkeys(np.clip(actions,0,1).tolist()))
            mse=[]
            for action in actions:
                budget.checkpoint()
                p,_=continuation_pnl(model,data,state,torch.full((count,),action),.0005,delay)
                mse.append(float(p.square().mean()))
            selected=actions[int(np.argmin(mse))]
        else:
            old,_=continuation_pnl(model,data,state,torch.full((count,),base),.0005,delay)
            new,_=continuation_pnl(model,data,state,torch.full((count,),selected),.0005,delay)
            accepted,record=accepted_action(new.numpy(),old.numpy())
            return dict(**record,baseline_action=base,selected_action=selected,accepted=accepted,
                        baseline_mse=float(old.square().mean()),selected_mse=float(new.square().mean()),
                        executed_mse=float((new if accepted else old).square().mean()),
                        selected_on_256=True,confirmation_is_decision_sample=True)


def diagnostic(project,out,end,budget):
    import torch
    from bottleneck_train import inputs_for
    torch.set_num_threads(1)
    task=dict(end=end,kind='plain');inputs=inputs_for(project,task,out)
    selected=[]
    for split,year in (('train',inputs.folds[0]),('validation',inputs.folds[1])):
        paths,data=inputs.observed(year)
        data=augment_real(paths,data,inputs,out,end)
        selected.extend((split,i,t,paths,data) for i,t in select_states(paths,data,split=='train'))
    manifest=[dict(number=n,split=split,path_index=i,cut=t,date=paths[i][t]['date'],episode=paths[i][t]['episode'])
              for n,(split,i,t,paths,data) in enumerate(selected)]
    manifest_file=out/'planner'/str(end)/'diagnostic_states.json'
    if manifest_file.exists() and json.loads(manifest_file.read_text())!=manifest:
        raise ValueError('Registered diagnostic states changed')
    write(manifest_file,manifest)
    results=[]
    for seed in range(3):
        model=model_for(out,end,seed)
        for number,(split,i,t,paths,data) in enumerate(selected):
            budget.checkpoint()
            record_path=out/'planner'/str(end)/'diagnostic'/f's{seed}_{number}.result.json'
            if record_path.exists():record=json.loads(record_path.read_text())
            else:
                with torch.no_grad():state=state_snapshot(model,data,paths,inputs,i,t,out,end)
                state_path=record_path.with_suffix('.state.json');write(state_path,state)
                with torch.no_grad():record=check_state(project,out,end,seed,state,model,state_path,budget)
                record.update(seed=seed,state_number=number,split=split,date=state['date'],episode=state['episode'])
                write(record_path,record)
            results.append(record)
    by_seed=[]
    for seed in range(3):
        rows=[r for r in results if r['seed']==seed]
        ref=np.mean([r['baseline_mse'] for r in rows]);new=np.mean([r['executed_mse'] for r in rows])
        by_seed.append(dict(seed=seed,relative_mse_improvement=float((ref-new)/max(ref,1e-20)),
                            accepted=sum(r['accepted'] for r in rows),states=len(rows)))
    promote=bool(sum(r['relative_mse_improvement']>0 for r in by_seed)>=2 and
                 np.mean([r['relative_mse_improvement'] for r in by_seed])>=.01 and any(r['accepted'] for r in by_seed))
    return dict(status='complete',promote=promote,states=64,results=len(results),by_seed=by_seed,
                limitation='Independent confirmation is used for action acceptance; not unbiased policy evaluation.')


def real_evaluation(project,out,end,stage,budget,cost_bp=5,delay=False):
    import torch
    import gzip
    import csv
    from torch import nn
    from bottleneck_train import inputs_for,job_for,evaluate
    from extended_train import predictions
    from contract_ledger import ledger,EVENTS
    from continuation_account import account_rollout
    from prepare_common import digest
    from train import expected_shortfall

    class FixedActions(nn.Module):
        def __init__(self,ratios):
            super().__init__();self.dummy=nn.Parameter(torch.tensor(0.));self.ratios=ratios
        def prepare(self,data):return self.ratios.T
        def action(self,prepared,previous,wealth):return prepared,None

    class Offset(nn.Module):
        wealth=True
        def __init__(self,model,offset):super().__init__();self.base=model;self.offset=offset
        def prepare(self,data):return self.base.prepare(data)
        def action(self,prepared,previous,wealth):
            return (self.base.action(prepared,previous,wealth)[0]+self.offset).clamp(0,1),None

    torch.set_num_threads(1)
    inputs=inputs_for(project,dict(end=end,kind='plain'),out)
    year=inputs.folds[1 if stage=='validate' else 2]
    paths,data=inputs.observed(year);data=augment_real(paths,data,inputs,out,end)
    folder=out/'planner'/str(end)/stage
    if stage=='evaluate':folder=folder/f'{cost_bp}bp_delay{int(delay)}'
    folder.mkdir(parents=True,exist_ok=True)
    fee=cost_bp/10000
    def measured(model):
        from fullstate_policy import FullStatePolicy
        for module in model.modules():
            if isinstance(module,FullStatePolicy):module.execution_delay=delay
        with torch.no_grad():p,r,s=account_rollout(model,data,fee,delay=delay)
        p=p.numpy()
        return dict(mse=float(np.mean(p*p)),rmse=float(np.sqrt(np.mean(p*p))),
                    cvar=float(expected_shortfall(-p)),cost=float(s['cost'].mean()),mean=float(p.mean()))
    summary=[];offset_records=[]
    for seed in range(3):
        model=model_for(out,end,seed,True)
        job=job_for(dict(end=end,seed=seed,kind='plain'))
        reference=measured(model)
        if stage=='validate':
            offsets=[]
            for offset in (-.15,-.05,0.,.05,.15):
                offsets.append(dict(offset=offset,**measured(Offset(model,offset))))
            best=min(offsets,key=lambda r:(r['rmse'],abs(r['offset'])))
            offset_records.append(dict(seed=seed,candidates=offsets,selected=best))
        else:
            chosen=json.loads((out/'planner'/str(end)/'validate.json').read_text())
            offset=next(r['selected']['offset'] for r in chosen['offset_controls'] if r['seed']==seed)
            best=dict(offset=offset,**measured(Offset(model,offset)))
            offset_records.append(dict(seed=seed,selected=best))
        rows=[]
        for i,path in enumerate(paths):
            budget.checkpoint()
            recordfile=folder/f's{seed}_path{i}.json'
            if recordfile.exists():
                record=json.loads(recordfile.read_text())
                if record['episode']!=path[0]['episode'] or record['source_sha256']!=digest(Path(__file__)):
                    raise ValueError('Stale planner path result retained')
                rows.append(record);continue
            # Preserve delayed absolute shares across unit changes in ledger precision.
            one={k:v[i:i+1] for k,v in data.items()};ratios=torch.zeros((1,one['active'].shape[1]),dtype=torch.float64)
            decisions=[];pending=0.
            for t in range(len(path)-1):
                budget.checkpoint()
                decisionfile=folder/f's{seed}_path{i}_t{t}.json'
                with torch.no_grad():state=state_snapshot(FixedActions(ratios),one,[path],inputs,0,t,out,end,fee)
                if delay:
                    # The prior signal is an absolute share instruction. Execute
                    # it first, then observe own post-trade account for the new signal.
                    traded=pending-state['shares'];charge=fee*abs(traded)*state['spot']
                    state['cash']-=traded*state['spot']+charge;state['shares']=pending
                    state['past_cost']+=charge/state['notional']
                    if one['record_event'][0,t]>0:
                        state['pending_asset']+=pending*float(one['record_amount'][0,t])
                    ratios[0,t]=pending/state['unit']
                state_path=decisionfile.with_suffix('.state.json')
                if decisionfile.exists():
                    previous_state=json.loads(state_path.read_text())
                    if previous_state!=state:raise ValueError('Cached planner decision state changed')
                    decision=json.loads(decisionfile.read_text())
                else:
                    write(state_path,state)
                    with torch.no_grad():decision=check_state(project,out,end,seed,state,model,state_path,budget,delay=delay)
                    write(decisionfile,decision)
                # Confirmation first actions use the neural float32 action grid;
                # keep that action exact, then execute absolute shares in float64.
                target=float(np.float32(decision['selected_action'] if decision['accepted'] else decision['baseline_action']))
                if delay:pending=target*state['unit']
                else:ratios[0,t]=target
                decisions.append(dict(date=path[t]['date'],ratio=float(ratios[0,t]),**decision))
            book=ledger(path,map(float,ratios[0,:len(path)-1]),float(path[0]['option_close']),fee)
            nominal=float(path[0]['spot'])*float(path[0]['unit'])
            pnl=float(book[-1]['wealth'])/nominal
            cost=float(sum(row['cost'] for row in book))/nominal
            turnover=float(sum(abs(row['traded'])*float(p['spot']) for row,p in zip(book,path)))/nominal
            with torch.no_grad():independent,_,_=account_rollout(FixedActions(ratios),one,fee)
            ledger_error=abs(float(independent[0])-pnl)
            if ledger_error>1e-8:raise AssertionError('Planner executed ledger fails independent continuation')
            record=dict(episode=path[0]['episode'],expiry=path[0]['expiry'],pnl=pnl,cost=cost,
                turnover=turnover,ledger_error=ledger_error,term=path[0]['term'],moneyness=path[0]['moneyness'],
                adjusted=any(int(p['unit'])!=10000 for p in path),
                cross_dividend=any(path[0]['date']<event[0]<=path[-1]['date'] for event in EVENTS),
                mean_ratio=float(ratios[0,:len(path)-1].mean()),accepted=sum(d['accepted'] for d in decisions),
                decisions=len(decisions),source_sha256=digest(Path(__file__)),
                weights_sha256=digest(out/'policies'/f'{end}_fullstate_s{seed}.pt'))
            with gzip.open(folder/f's{seed}_path{i}.daily.csv.gz','wt',encoding='utf8',newline='') as file:
                writer=csv.DictWriter(file,fieldnames=book[0].keys());writer.writeheader();writer.writerows(book)
            write(recordfile,record);rows.append(record)
        if len(rows)!=len(paths):raise AssertionError('Planner evaluation omitted paths')
        p=np.array([r['pnl'] for r in rows]);c=np.array([r['cost'] for r in rows])
        summary.append(dict(seed=seed,paths=len(rows),rmse=float(np.sqrt(np.mean(p*p))),
            cvar=float(expected_shortfall(-p)),cost=float(c.mean()),mean=float(p.mean()),
            reference=reference,best_offset=best,accepted=sum(r['accepted'] for r in rows)))
        if stage=='evaluate':
            key=f'{end}_planner_s{seed}';name=f'{key}_{cost_bp}bp_delay{int(delay)}'
            daily=out/'evaluation/daily'/f'{name}.csv.gz';daily.parent.mkdir(parents=True,exist_ok=True)
            first=True
            with gzip.open(daily,'wt',encoding='utf8',newline='') as stream:
                for i in range(len(paths)):
                    with gzip.open(folder/f's{seed}_path{i}.daily.csv.gz','rt',encoding='utf8',newline='') as part:
                        header=next(part)
                        if first:stream.write(header);first=False
                        for line in part:stream.write(line)
            metrics=dict(key=key,end=end,kind='planner',seed=seed,cost_bp=cost_bp,delay=delay,n=len(paths),
                rmse=float(np.sqrt(np.mean(p*p))),cvar=float(expected_shortfall(-p)),mean=float(p.mean()),
                std=float(p.std()),cost=float(c.mean()),turnover=float(np.mean([r['turnover'] for r in rows])),
                ledger_error=max(r['ledger_error'] for r in rows))
            write(out/'evaluation/paths'/f'{name}.json',dict(metrics=metrics,
                weights_sha256=digest(out/'policies'/f'{end}_fullstate_s{seed}.pt'),daily_sha256=digest(daily),
                rows=[{key:r[key] for key in ('episode','pnl','expiry','term','moneyness','adjusted','cost','turnover','cross_dividend')}
                      for r in rows]))
    mean=lambda name:float(np.mean([r[name] for r in summary]))
    ref=lambda name:float(np.mean([r['reference'][name] for r in summary]))
    promote=bool(mean('rmse')<=.99*ref('rmse') and mean('cvar')<=1.05*ref('cvar') and
                 mean('cost')<=1.05*ref('cost') and sum(r['rmse']<r['reference']['rmse'] for r in summary)>=2 and
                 mean('rmse')<np.mean([r['best_offset']['rmse'] for r in summary]))
    return dict(status='complete',promote=promote,by_seed=summary,offset_controls=offset_records,
        paths=len(paths),path_ids=[p[0]['episode'] for p in paths],cost_basis_points=cost_bp,delay=delay,
        observed_historical_evaluation=True,simulation='current terms frozen, no future dividends')


def all_pressures(project,out,end,budget):
    from bottleneck_train import inputs_for
    base=inputs_for(project,dict(end=end,kind='plain'),out)
    full=FullInputs(base,out,end)
    controls=[]
    for seed in range(3):
        budget.checkpoint()
        task=dict(end=end,kind='fullstate',seed=seed,key=f'{end}_fullstate_s{seed}')
        controls.extend(control_pressures(project,out,task,full,budget))
    scenes=[]
    for bp in (0,5,10):
        for delay in (False,True):
            budget.checkpoint()
            scene=real_evaluation(project,out,end,'evaluate',budget,bp,delay)
            write(out/'planner'/str(end)/f'evaluate_{bp}bp_delay{int(delay)}.json',scene)
            scenes.append(scene)
    return dict(status='complete',scenarios=scenes,control_metrics=controls,paths=scenes[0]['paths'],
                path_ids=scenes[0]['path_ids'],execution_scenarios=6)


def control_pressures(project,out,task,inputs,budget):
    import torch,gzip,csv
    from continuation_account import account_rollout
    from extended_data import books
    from bottleneck_core import costs
    from contract_ledger import EVENTS
    from prepare_common import digest
    from train import expected_shortfall
    end,seed=task['end'],task['seed'];model=model_for(out,end,seed,True)
    weights=out/'policies'/f'{task["key"]}.pt';paths,data=inputs.observed(end);result=[]
    for bp in (0,5,10):
        for delay in (False,True):
            budget.checkpoint();model.execution_delay=delay
            with torch.no_grad():p,r,state=account_rollout(model,data,bp/10000,delay=delay)
            pnl=p.numpy();ratios=r.numpy();fees=state['cost'].numpy()
            turnover=costs(r,data,1.).detach().numpy()
            ledgers=books(paths,ratios,None,bp)
            independently=np.array([b[-1]['wealth']/(float(path[0]['spot'])*int(path[0]['unit']))
                                    for path,b in zip(paths,ledgers)])
            error=float(np.max(abs(pnl-independently)))
            if error>1e-8:raise AssertionError('Fullstate stress ledger mismatch')
            name=f'{task["key"]}_{bp}bp_delay{int(delay)}'
            daily=out/'evaluation/daily'/f'{name}.csv.gz';daily.parent.mkdir(parents=True,exist_ok=True)
            with gzip.open(daily,'wt',encoding='utf8',newline='') as file:
                writer=csv.DictWriter(file,fieldnames=ledgers[0][0].keys());writer.writeheader()
                for book in ledgers:writer.writerows(book)
            metrics=dict(key=task['key'],end=end,kind='fullstate',seed=seed,cost_bp=bp,delay=delay,n=len(paths),
                rmse=float(np.sqrt(np.mean(pnl*pnl))),cvar=float(expected_shortfall(-pnl)),mean=float(pnl.mean()),
                std=float(pnl.std()),cost=float(fees.mean()),turnover=float(turnover.mean()),ledger_error=error)
            write(out/'evaluation/paths'/f'{name}.json',dict(metrics=metrics,weights_sha256=digest(weights),
                daily_sha256=digest(daily),rows=[dict(episode=path[0]['episode'],pnl=float(z),expiry=path[0]['expiry'],
                    term=path[0]['term'],moneyness=path[0]['moneyness'],adjusted=any(int(row['unit'])!=10000 for row in path),
                    cost=float(f),turnover=float(v),cross_dividend=any(path[0]['date']<e[0]<=path[-1]['date'] for e in EVENTS))
                    for path,z,f,v in zip(paths,pnl,fees,turnover)]))
            result.append(metrics)
    return result


def main(args):
    global BRIDGE
    project=args.project.resolve();out=args.out.resolve();setup(project)
    if args.stage=='generate':return generate(project,out,args.end,args.state,args.count,args.phase)
    if args.stage=='raw':return export_raw(project,out,args.end)
    if args.stage=='serve':return serve(project,out,args.end,args.queue)
    if args.stage=='fit':
        class ParentBudget:
            used=0
            def checkpoint(self):pass
        return train_fullstate(project,out,args.end,ParentBudget(),only_seed=args.seed)
    folder=out/'planner'/str(args.end);folder.mkdir(exist_ok=True,parents=True)
    lock=out/'planner/.queue.lock'
    fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    budget=Budget(out)
    try:
        from prepare_common import digest
        policy_kind='plain' if args.stage=='diagnostic' else 'fullstate'
        dependencies=[Path(__file__),Path(__file__).with_name('fullstate_policy.py'),
            Path(__file__).with_name('continuation_account.py'),
            out/'markets'/str(args.end)/'reference_coefficients.npz',
            project/'工作区/数据/处理后/创新组合实验'/str(args.end)/'episode_days.csv',
            project/'工作区/数据/处理后/创新组合实验'/str(args.end)/'C1/common.npz']
        if args.stage!='train':
            dependencies += [out/'policies'/f'{args.end}_{policy_kind}_s{seed}.pt' for seed in range(3)]
        identity=dict(files={str(p):digest(p) for p in dependencies})
        stamp=folder/f'{args.stage}.identity.json'
        if stamp.exists() and json.loads(stamp.read_text())!=identity:
            raise ValueError('Planner source or frozen parent changed; old results preserved')
        write(stamp,identity)
        result_path=folder/f'{args.stage}.json'
        if result_path.exists() and json.loads(result_path.read_text()).get('status')=='complete':
            return
        budget.checkpoint()
        if args.stage in ('diagnostic','validate','evaluate'):BRIDGE=MarketBridge(project,out,args.end,budget)
        if args.stage=='diagnostic':result=diagnostic(project,out,args.end,budget)
        elif args.stage=='train':result=train_fullstate(project,out,args.end,budget)
        elif args.stage=='validate':result=real_evaluation(project,out,args.end,args.stage,budget)
        elif args.stage=='evaluate':result=all_pressures(project,out,args.end,budget)
        else:raise NotImplementedError('Stage implementation not yet installed')
        write(folder/f'{args.stage}.json',result)
    except BudgetExceeded as e:
        write(folder/f'{args.stage}.json',dict(status='budget_incomplete',reason=str(e)))
    except subprocess.TimeoutExpired:
        budget.used=86400
        write(folder/f'{args.stage}.json',dict(status='budget_incomplete',reason='Remaining planning time exhausted'))
    except Exception as e:
        write(folder/f'{args.stage}.json',dict(status='failed',reason=repr(e)));raise
    finally:
        if BRIDGE is not None:BRIDGE.close();BRIDGE=None
        try:budget.checkpoint()
        except BudgetExceeded:pass
        lock.unlink(missing_ok=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--project',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--end',type=int,required=True)
    p.add_argument('--stage',choices=['diagnostic','train','validate','evaluate','generate','raw','fit','serve'],required=True)
    p.add_argument('--state',type=Path);p.add_argument('--count',type=int);p.add_argument('--phase',type=int)
    p.add_argument('--seed',type=int)
    p.add_argument('--queue',type=Path)
    main(p.parse_args())
