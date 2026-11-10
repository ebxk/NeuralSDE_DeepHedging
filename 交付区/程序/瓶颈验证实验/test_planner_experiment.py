import unittest
import numpy as np
import torch
import bootstrap
from fullstate_policy import FullStatePolicy
from planner_experiment import accepted_action, continuation_pnl

class PlannerTests(unittest.TestCase):
    def test_acceptance_keeps_all_confirmation_paths(self):
        ok,record=accepted_action(np.full(1024,.005),np.full(1024,.01))
        self.assertTrue(ok)
        self.assertAlmostEqual(record['relative_mse_improvement'],.75)
        self.assertFalse(accepted_action(np.full(1024,.011),np.full(1024,.01))[0])

    def test_account_continuation_keeps_original_cash_and_notional(self):
        model=FullStatePolicy()
        for p in model.parameters():p.data.zero_()
        data=dict(f=torch.zeros(2,1,8),return_fifo=torch.zeros(2,1,20),raw_ewma=torch.ones(2,1),
                  delta=torch.zeros(2,1),delta_valid=torch.zeros(2,1),notional_scale=torch.ones(2,1),
                  spot=torch.tensor([[2.,3.],[2.,3.]],dtype=torch.float64),active=torch.ones(2,1,dtype=torch.bool))
        state=dict(cash=1.,shares=0.,receivable=.2,pending_asset=.1,notional=20.,unit=10.,strike=3.)
        pnl,cost=continuation_pnl(model,data,state,torch.tensor([0.,1.]),0.)
        np.testing.assert_allclose(pnl.numpy(),[1.3/20,11.3/20])
        delayed,_=continuation_pnl(model,data,state,torch.tensor([0.,1.]),.001,delay=True)
        np.testing.assert_allclose(delayed.numpy(),[1.3/20,1.3/20])

    def test_fullstate_wealth_and_position_gradients_survive(self):
        model=FullStatePolicy();prev=torch.ones(2,requires_grad=True);wealth=torch.ones(2,requires_grad=True)
        p=model.prepare(dict(f=torch.zeros(2,1,8),return_fifo=torch.zeros(2,1,20),raw_ewma=torch.ones(2,1),
                            delta=torch.zeros(2,1),delta_valid=torch.zeros(2,1),notional_scale=torch.ones(2,1)))
        a=model.action(p[0],prev,wealth)[0]
        grads=torch.autograd.grad(a.sum(),(prev,wealth))
        self.assertTrue(all(torch.isfinite(g).all() and (g!=0).any() for g in grads))

    def test_control_observes_same_delta_validity_and_original_notional_scale(self):
        model=FullStatePolicy()
        self.assertEqual(model.network[0].in_features,35)
        self.assertEqual(sum(p.numel() for p in model.parameters()),3865)
        for p in model.parameters():p.data.zero_()
        model.network[0].weight.data[0,29:32]=1
        data=dict(f=torch.zeros(2,1,8),return_fifo=torch.zeros(2,1,20),raw_ewma=torch.ones(2,1),
                  delta=torch.tensor([[.2],[.8]]),delta_valid=torch.tensor([[1.],[0.]]),
                  notional_scale=torch.tensor([[1.1],[.9]]))
        torch.testing.assert_close(model.prepare(data)[0,:,0],torch.tensor([2.3,1.7]))

    def test_delayed_maturity_cancels_order_and_cost_changes_closed_loop(self):
        from continuation_account import account_rollout
        model=FullStatePolicy()
        for p in model.parameters():p.data.zero_()
        # Wealth sensitivity makes higher executed fees change tomorrow's signal.
        model.network[0].weight.data[0,-2]=.05
        model.network[2].weight.data[0,0]=1;model.network[4].weight.data[0,0]=1
        n,h=1,3
        data=dict(f=torch.zeros(n,h,8),return_fifo=torch.zeros(n,h,20),raw_ewma=torch.ones(n,h),
            delta=torch.zeros(n,h),delta_valid=torch.zeros(n,h),notional_scale=torch.ones(n,h),
            spot=torch.tensor([[2.,2.2,2.4,2.6]],dtype=torch.float64),unit=torch.tensor([[10.,20.,20.,20.]],dtype=torch.float64),
            active=torch.ones(n,h,dtype=torch.bool),premium=torch.tensor([.1]),strike_end=torch.tensor([3.]))
        with torch.no_grad():
            p0,r0,s0=account_rollout(model,data,0.,delay=True)
            p1,r1,s1=account_rollout(model,data,.001,delay=True)
        self.assertEqual(float(r0[0,0]),0)
        # Prior day's absolute shares execute; unit doubling does not double shares.
        self.assertLess(float(r0[0,1]),.5)
        self.assertNotEqual(float(r0[0,2]),float(r1[0,2]))
        self.assertEqual(float(s1['shares'][0]),0.)
        self.assertEqual(float(s1['pending'][0]),0.)
        self.assertGreater(float(s1['cost'][0]),0.)
        self.assertLess(float(p1[0]),float(p0[0]))

    def test_own_pending_rights_record_ex_and_forced_action_keep_gradient(self):
        model=FullStatePolicy()
        for p in model.parameters():p.data.zero_()
        model.network[0].weight.data[0,32]=1
        model.network[2].weight.data[0,0]=1;model.network[4].weight.data[0,0]=1
        data=dict(f=torch.zeros(1,3,8),return_fifo=torch.zeros(1,3,20),raw_ewma=torch.ones(1,3),
            delta=torch.zeros(1,3),delta_valid=torch.zeros(1,3),notional_scale=torch.ones(1,3),
            initial_notional=torch.tensor([20.]),unit=torch.tensor([[10.,10.,20.,20.]]),
            record_amount=torch.tensor([[.1,0.,0.,0.]]),ex_event=torch.tensor([[0,0,1,0]]))
        prepared=model.prepare(data)
        first=model.action(prepared[0],torch.zeros(1),torch.zeros(1))[0]
        second=model.action(prepared[1],first,torch.zeros(1))[0]
        third=model.action(prepared[2],second/2,torch.zeros(1))[0]
        self.assertAlmostEqual(float(second.detach()),float(torch.sigmoid(torch.tanh(torch.tanh(torch.tensor(.025))))))
        self.assertAlmostEqual(float(third.detach()),.5)
        self.assertGreater(float(torch.autograd.grad(second,first,retain_graph=True)[0]),0.)
        prepared=model.prepare(data)
        forced_shares=torch.tensor([3.],requires_grad=True)
        forced_next=model.action(prepared[1],forced_shares/10,torch.zeros(1))[0]
        self.assertGreater(float(forced_next.detach()),.5)
        self.assertGreater(float(torch.autograd.grad(forced_next,forced_shares)[0]),0.)
        model.execution_delay=True;prepared=model.prepare(data)
        model.action(prepared[0],torch.zeros(1),torch.zeros(1))
        delayed_next=model.action(prepared[1],torch.tensor([.5]),torch.zeros(1))[0]
        self.assertAlmostEqual(float(delayed_next.detach()),.5)

if __name__=='__main__':unittest.main()
