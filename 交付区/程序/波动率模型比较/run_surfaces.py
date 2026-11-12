"""Resumable full/held-quote calibration; failures never remove hedge paths."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time
import traceback
import numpy as np
import pandas as pd
from study_core import HERE, OUT, CONFIG, digest, write_json
from volatility_surfaces import fit_surface, price_surface, summary

def calibrate(task):
    day,kind,view,records=task
    out=OUT/'surfaces'/kind/day/f'{view}.json'
    identity=dict(source=digest(HERE/'volatility_surfaces.py'),
        runner=digest(HERE/'run_surfaces.py'),quotes=digest(OUT/'data/quotes.csv'),
        day=day,kind=kind,view=view,seed=CONFIG['calibration_seed'],starts=5)
    if out.exists():
        old=json.loads(out.read_text(encoding='utf8'))
        if old['identity']==identity:return dict(day=day,kind=kind,view=view,status='cached')
        raise ValueError(f'Changed calibration identity: {out}; preserve old study before rerun')
    q=pd.DataFrame(records);visible=~q.holdout if view=='heldout' else np.ones(len(q),bool)
    try:
        model=fit_surface(kind,q.ttm.to_numpy()[visible],q.k.to_numpy()[visible],
            q.price.to_numpy()[visible],seed=CONFIG['calibration_seed'],starts=5)
        predictions=price_surface(model,q.ttm.to_numpy(),q.k.to_numpy())
        errors=predictions-q.price.to_numpy()
        stats={}
        for name,mask in [('visible',visible),('heldout',q.holdout.to_numpy())]:
            values=errors[mask]
            stats[name]=dict(n=len(values),rmse=float(np.sqrt(np.mean(values**2))) if len(values) else None)
        record=dict(identity=identity,model=model,summary=summary(model),metrics=stats,
            status='complete' if model['diagnostics']['valid'] else 'numerical_rejected')
    except Exception:
        record=dict(identity=identity,status='failed',reason=traceback.format_exc())
    write_json(out,record)
    return dict(day=day,kind=kind,view=view,status=record['status'])

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--days',type=int,default=0,help='0 means every date; partial runs are never complete')
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    q=pd.read_csv(OUT/'data/quotes.csv');groups=list(q.groupby('date',sort=True))
    selected=groups[:args.days] if args.days else groups
    tasks=[(day,kind,view,g.to_dict('records')) for day,g in selected
           for kind in ('svi','ssvi','essvi') for view in ('full','heldout')]
    start=time.monotonic();done=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(calibrate,t) for t in tasks]
        for future in as_completed(futures):
            result=future.result();done.append(result)
            if len(done)%6==0:
                status=dict(phase='surface_calibration',finished=len(done),expected=len(tasks),
                    study_dates=len(groups),selected_dates=len(selected),complete=False,
                    elapsed_seconds=time.monotonic()-start,last=result,
                    remaining_stages=['direct_hedges','dynamic_markets','simulation',
                        'hedge_training_432','evaluation','delivery'])
                write_json(OUT/'pipeline_status.json',status)
                print(json.dumps(status,ensure_ascii=False),flush=True)
    write_json(OUT/'surface_stage.json',dict(stage_complete=not args.days,
        expected=len(tasks),results=done,seconds=time.monotonic()-start,
        scope='SVI/SSVI/eSSVI full and held-quote calibration; not whole study'))

if __name__=='__main__':
    try:main()
    except BaseException:
        write_json(OUT/'surface_producer_failure.json',dict(complete=False,reason=traceback.format_exc()))
        raise
