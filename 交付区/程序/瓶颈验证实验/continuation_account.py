"""Exact account continuation and frozen-policy execution stresses."""
import torch
from extended_policy import _at

def account_rollout(model,data,fee,initial=None,stop=None,delay=False):
    spot,unit,active=data['spot'],data['unit'],data['active'].bool()
    n,h=active.shape
    zeros=torch.zeros(n,dtype=spot.dtype,device=spot.device)
    record,ex,pay=(data.get(k) for k in ('record_event','ex_event','pay_event'))
    ids=sorted(set(record[record>0].tolist())) if record is not None else []
    if initial is None:
        cash=data['premium'].to(spot.dtype)*unit[:,0]
        shares=zeros.clone(); receivable=zeros.clone(); total_cost=zeros.clone()
        notional=spot[:,0]*unit[:,0]
        entitlements={k:zeros.clone() for k in ids}
        settled=torch.zeros(n,dtype=torch.bool,device=spot.device)
        pending=zeros.clone(); start=0
    else:
        cash,shares,receivable,notional=(initial[k] for k in ('cash','shares','receivable','notional'))
        total_cost=initial['cost']; pending=initial['pending']; settled=initial['settled']
        entitlements=dict(initial['entitlements']); start=initial['next_index']
        for k in ids:
            entitlements.setdefault(k,zeros.clone())
    prepared=model.prepare(data)
    ratios=[]
    end=h+1 if stop is None else stop
    for t in range(start,end):
        if entitlements:
            accrued=sum(torch.where(ex[:,t]==key,amount,zeros) for key,amount in entitlements.items()) if ex is not None else 0
            paid=sum(torch.where(pay[:,t]==key,amount,zeros) for key,amount in entitlements.items()) if pay is not None else 0
            receivable=receivable+accrued-paid; cash=cash+paid
        decision=active[:,t] if t<h else torch.zeros(n,dtype=torch.bool,device=spot.device)
        settlement=~decision & ~settled
        dtype=next(model.parameters()).dtype
        if t<h and not delay:
            wealth=(cash+receivable+shares*spot[:,t])/notional
            ratio,_=model.action(_at(prepared,t),shares.to(dtype)/unit[:,t].to(dtype),wealth)
            target=torch.where(decision,ratio.to(spot.dtype)*unit[:,t],zeros)
        else:
            target=torch.where(decision,pending,zeros) if delay else zeros
        traded=target-shares
        cost=fee*traded.abs()*spot[:,t]
        total_cost=total_cost+cost/notional
        cash=cash-traded*spot[:,t]-cost
        cash=cash-torch.where(settlement,unit[:,t]*torch.relu(spot[:,t]-data['strike_end']),zeros)
        shares=target
        if t<h:
            ratios.append(shares/unit[:,t])
            if delay:
                wealth=(cash+receivable+shares*spot[:,t])/notional
                signal,_=model.action(_at(prepared,t),shares.to(dtype)/unit[:,t].to(dtype),wealth)
                pending=torch.where(decision,signal.to(spot.dtype)*unit[:,t],zeros)
        for key in entitlements:
            if record is not None:
                entitlements[key]=torch.where(record[:,t]==key,shares*data['record_amount'][:,t],entitlements[key])
        settled=settled|settlement
        pending=torch.where(settled,zeros,pending)
    state=dict(cash=cash,shares=shares,receivable=receivable,notional=notional,
               entitlements=entitlements,settled=settled,pending=pending,cost=total_cost,next_index=end)
    pnl=(cash+receivable+shares*spot[:,end-1])/notional
    return pnl,torch.stack(ratios,1) if ratios else spot[:,:0],state
