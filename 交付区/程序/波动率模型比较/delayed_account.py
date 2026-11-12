"""Today submits absolute shares; next close fills before the next decision."""
import torch

def delayed_rollout(model,data,fee):
    s,u,a=data['spot'],data['unit'],data['active'].bool();n,h=a.shape
    cash=data['premium']*u[:,0];notional=s[:,0]*u[:,0]
    shares=torch.zeros_like(cash);receivable=torch.zeros_like(cash);pending=torch.zeros_like(cash)
    settled=torch.zeros(n,dtype=torch.bool);prepared=model.prepare(data);ratios=[]
    record=data.get('record_event');ex=data.get('ex_event');pay=data.get('pay_event')
    ids=sorted(set(record[record>0].tolist())) if record is not None else []
    rights={i:torch.zeros_like(cash) for i in ids}
    for t in range(h+1):
        for event,amount in rights.items():
            accrued=torch.where(ex[:,t]==event,amount,torch.zeros_like(amount))
            paid=torch.where(pay[:,t]==event,amount,torch.zeros_like(amount))
            receivable=receivable+accrued-paid;cash=cash+paid
        decision=a[:,t] if t<h else torch.zeros_like(settled)
        settlement=~decision & ~settled
        target=torch.where(decision,pending,torch.zeros_like(pending))
        traded=target-shares;cash=cash-traded*s[:,t]-fee*traded.abs()*s[:,t]
        payoff=u[:,t]*torch.relu(s[:,t]-data['strike_end'])
        cash=cash-torch.where(settlement,payoff,torch.zeros_like(payoff))
        shares=target;settled=settled|settlement
        for event in ids:
            rights[event]=torch.where(record[:,t]==event,shares*data['record_amount'][:,t],rights[event])
        if t<h:
            ratios.append(shares/u[:,t])
            action,_=model.action(prepared[t],shares/u[:,t],(cash+receivable+shares*s[:,t])/notional)
            pending=torch.where(decision,action*u[:,t],torch.zeros_like(pending))
    return (cash+receivable+shares*s[:,-1])/notional,torch.stack(ratios,dim=1)
