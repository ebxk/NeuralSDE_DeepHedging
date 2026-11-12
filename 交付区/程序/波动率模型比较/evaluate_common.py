"""Historical evaluation is separate from checkpoint selection."""
import json
import time
import numpy as np
import pandas as pd
import torch
from study_core import OUT,CONFIG,registry,write_json,digest,bootstrap_legacy
from common_policy import CommonPolicy
from train_common import load_data,subset

def crosses_ex_date(start,end,events):return any(start<event[0]<=end for event in events)

def run():
    torch.set_num_threads(2)
    bootstrap_legacy();from extended_policy import rollout
    from train import expected_shortfall
    from hedge_experiment import read_paths
    from contract_ledger import ledger,EVENTS
    from delayed_account import delayed_rollout
    from study_core import PROJECT
    tasks=registry();completed=[];unavailable=[]
    for j in tasks:
        r=OUT/'policies'/f"{j['key']}.json"
        if r.exists():completed.append(j)
        elif (OUT/'failures'/f"{j['key']}.json").exists():unavailable.append(j)
        else:raise ValueError(f"Training selection is not finished: {j['key']}")
    # Freeze selections before any evaluation-date PnL is generated.
    selections={j['key']:digest(OUT/'policies'/f"{j['key']}.json") for j in completed}
    from study_core import frozen_json
    frozen_json(OUT/'selected_checkpoints.json',dict(selections=selections,unavailable=unavailable))
    root=OUT/'evaluation';root.mkdir(exist_ok=True);records=[]
    for job in completed:
        inp=OUT/f"inputs/{job['end']}";scale=dict(np.load(inp/'feature_scale.npz'))
        data=load_data(inp/f"real_{job['end']}.npz",**scale)
        model=CommonPolicy(job['information']=='surface')
        weights=OUT/'policies'/f"{job['key']}.pt"
        model.load_state_dict(torch.load(weights,weights_only=True))
        receipt=json.loads((OUT/'policies'/f"{job['key']}.json").read_text(encoding='utf8'))
        from artifact_guard import check_hashes,verify_ready
        check_hashes(receipt['identity']['parents'])
        verify_ready(inp/f"sim_{job['model']}.ready.json",inp/f"sim_{job['model']}.npz",CONFIG,job['model'],job['end'])
        if digest(weights)!=receipt['weight_sha256']:raise ValueError('Weight changed')
        paths=read_paths(PROJECT/f"工作区/数据/处理后/创新组合实验/{job['end']}/episode_days.csv",[job['end']])[0]
        for cost,execution in ((0,'same_close'),(5,'same_close'),(10,'same_close'),(5,'next_close')):
            started=time.perf_counter()
            pnls=[];ratios=[]
            with torch.no_grad():
                for i in range(0,len(data['f']),128):
                    if execution=='same_close':
                        pnl,ratio,_=rollout(model,subset(data,slice(i,i+128)),cost/10000,capture=True)
                    else:pnl,ratio=delayed_rollout(model,subset(data,slice(i,i+128)),cost/10000)
                    pnls.extend(pnl.numpy());ratios.extend(ratio.numpy())
            p=np.array(pnls)
            replay_seconds=time.perf_counter()-started
            tag=f"{job['key']}_cost{cost}_{execution}"
            np.savez_compressed(root/f'{tag}.npz',pnl=p,ratios=ratios)
            daily=[];costs=[];turnover=[];path_rows=[];errors=[]
            for i,path in enumerate(paths):
                book=ledger(path,map(float,ratios[i][:len(path)-1]),float(path[0]['option_close']),cost/10000)
                nominal=float(path[0]['spot'])*int(path[0]['unit'])
                independent=book[-1]['wealth']/nominal
                errors.append(abs(independent-p[i]))
                c=sum(row['cost'] for row in book)/nominal
                v=sum(abs(row['traded'])*float(market['spot']) for row,market in zip(book,path))/nominal
                costs.append(c);turnover.append(v)
                path_rows.append(dict(episode=path[0]['episode'],expiry=path[0]['expiry'],
                    month=path[0]['expiry'][:7],term=path[0]['term'],moneyness=path[0]['moneyness'],
                    cross_dividend=int(crosses_ex_date(path[0]['date'],path[-1]['date'],EVENTS)),
                    adjusted=int(any(int(x['unit'])!=10000 for x in path)),
                    pnl=p[i],cost=c,turnover=v))
                daily.extend(dict(episode=path[0]['episode'],**row) for row in book)
            if max(errors)>1e-8:raise ValueError(f'Independent ledger mismatch: {max(errors)}')
            pd.DataFrame(daily).to_csv(root/f'{tag}.ledger.csv.gz',index=False)
            pd.DataFrame(path_rows).to_csv(root/f'{tag}.paths.csv',index=False)
            records.append(dict(**job,cost_bp=cost,rmse=float(np.sqrt(np.mean(p*p))),
                cvar=float(expected_shortfall(-p)),mean=float(p.mean()),std=float(p.std()),
                cost=float(np.mean(costs)),turnover=float(np.mean(turnover)),ledger_max_error=max(errors),
                inference_parameters=sum(p.numel() for p in model.parameters()),training_seconds=receipt['seconds'],
                replay_seconds=replay_seconds,
                n=len(p),execution=execution,cost_scenario='reoptimized' if cost==5 and execution=='same_close' else 'frozen_weight_pressure'))
            write_json(root/'progress.json',dict(complete=False,cases_done=len(records),cases_expected=4*len(completed),last=tag))
    pd.DataFrame(records).to_csv(root/'overall.csv',index=False)
    write_json(root/'status.json',dict(complete=False,trained=len(completed),unavailable=len(unavailable),
        remaining=['groups','paired_month_bootstrap','conditional_prediction','reload_parity','plots_and_delivery']))

if __name__=='__main__':run()
