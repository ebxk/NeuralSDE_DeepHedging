"""Check the study's high-risk numerical and accounting boundaries."""
import math
import numpy as np
import torch

from contract_ledger import ledger
from policy import MLP,Blend,rollout
from train import adversary,expected_shortfall,pnl_sim


def main():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    f=torch.randn(4,3,8)
    units=torch.tensor([[10000.,10200.,10200.]]).repeat(4,1)
    base,factor=MLP(3),MLP(5)
    for model,w in [(base,0.),(factor,1.)]:
        mixed=Blend(base,factor,"fixed",w)
        torch.testing.assert_close(rollout(mixed,f,units)[0],rollout(model,f,units)[0],rtol=0,atol=0)
    gate=Blend(base,factor,"reliable_gate")
    prev=torch.tensor([.2,.4,.6,.8],requires_grad=True)
    grad=torch.autograd.grad(gate.step(f[:,1],prev).sum(),prev)[0]
    assert grad.abs().sum()>0
    assert not any(p.requires_grad for p in gate.base.parameters())
    sim=torch.tensor([[1.,1.03,.98,1.06]]).repeat(4,1)
    mask=torch.ones(4,3,dtype=torch.bool)
    val=pnl_sim(gate,f,sim,mask,torch.zeros(4),.0005)
    val.square().sum().backward()
    assert gate.gate[-2].weight.grad.abs().sum()>0
    risks=torch.tensor([1.,4.,2.],requires_grad=True)
    q=adversary(risks.detach(),0)
    (risks*torch.as_tensor(q,dtype=risks.dtype)).sum().backward()
    torch.testing.assert_close(risks.grad,torch.full((3,),1/3))
    q=adversary(risks.detach(),.1)
    assert abs(q.sum()-1)<1e-12 and abs(np.sum(q*np.log(q*3))-.1)<1e-6
    assert q[1]>q[2]>q[0] and q@risks.detach().numpy()>risks.mean().item()
    np.testing.assert_allclose(adversary([1,1,1]),[1/3]*3)
    # Worst 5% of 30 losses includes the maximum and half of the second worst.
    assert abs(expected_shortfall(np.arange(30))-(29+28/2)/1.5)<1e-12
    # Record-date shares earn the cash dividend even if the hedge ratio and unit change.
    events=[("2024-12-02","2024-11-29","2024-12-05",".05",10200)]
    rows=[dict(date=d,spot=s,unit=u,strike=k) for d,s,u,k in [
        ("2024-11-29",2.2,10000,2.1),("2024-12-02",2.15,10200,2.06),
        ("2024-12-05",2.21,10200,2.06),("2024-12-06",2.24,10200,2.06)]]
    book=ledger(rows,[.5,.5,.6],.10,.0005,events)
    assert book[1]["traded"]==100 and book[1]["accrued"]==250 and book[1]["receivable"]==250
    assert book[2]["paid"]==250 and book[2]["receivable"]==0
    assert book[-1]["shares"]==0 and math.isclose(book[-1]["liability"],1836)
    trades=[5000,100,1020,-6120]
    costs=sum(abs(a)*r["spot"]*.0005 for a,r in zip(trades,rows))
    expected=1000+250-sum(a*r["spot"] for a,r in zip(trades,rows))-1836-costs
    assert math.isclose(book[-1]["wealth"],expected)
    print("Gate endpoints and gradients, KL, fractional CVaR and cash ledger verified.")


if __name__=="__main__":
    main()

