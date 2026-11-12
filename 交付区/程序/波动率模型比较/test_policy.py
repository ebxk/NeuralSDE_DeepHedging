import unittest
import torch
from common_policy import CommonPolicy

class PolicyTests(unittest.TestCase):
    def test_base_cannot_read_surface_slots(self):
        torch.manual_seed(0)
        model=CommonPolicy(surface=False)
        a={'f':torch.zeros(2,3,7)};b={'f':torch.zeros(2,3,7)};b['f'][...,3:]=99
        x=model.prepare(a);y=model.prepare(b)
        torch.testing.assert_close(x,y)

    def test_previous_position_has_cross_period_gradient(self):
        torch.manual_seed(0);model=CommonPolicy(surface=True)
        p=torch.tensor([.4,.6],requires_grad=True)
        first=model.prepare({'f':torch.zeros(2,1,7)})[0]
        action,_=model.action(first,p,torch.zeros(2))
        action.sum().backward()
        self.assertGreater(float(p.grad.abs().sum()),0)

    def test_same_seed_has_same_weights_for_both_information_sets(self):
        torch.manual_seed(1);a=CommonPolicy(False)
        torch.manual_seed(1);b=CommonPolicy(True)
        for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q)

    def test_actual_simulation_ledger_has_one_entry_and_one_exit_fee(self):
        from study_core import bootstrap_legacy
        bootstrap_legacy()
        from extended_policy import simulated_pnl
        model=CommonPolicy(False)
        for p in model.parameters():p.data.zero_()
        data=dict(f=torch.zeros(1,2,7),spot=torch.tensor([[1.,1.1,1.05]],dtype=torch.float64),
            active=torch.ones(1,2,dtype=torch.bool),premium=torch.tensor([.1]),strike_end=torch.tensor([1.]))
        pnl=simulated_pnl(model,data,.001)
        self.assertAlmostEqual(float(pnl[0].detach()),.075-.001025,places=7)

if __name__=='__main__':unittest.main()
