import bootstrap
import unittest
import torch
from extended_policy import make_policy, rollout
from continuation_account import account_rollout

class AccountTests(unittest.TestCase):
    def test_resume_preserves_dividend_receivable_unit_adjustment(self):
        torch.manual_seed(7)
        model=make_policy('wide',wealth=True)
        data=dict(f=torch.zeros(1,3,8),spot=torch.tensor([[2.,1.9,2.,2.1]],dtype=torch.float64),
                  unit=torch.tensor([[10.,10.,11.,11.]],dtype=torch.float64),
                  premium=torch.tensor([.1]),strike_end=torch.tensor([2.]),
                  active=torch.ones(1,3,dtype=torch.bool),
                  record_event=torch.tensor([[1,0,0,0]]),ex_event=torch.tensor([[0,1,0,0]]),
                  pay_event=torch.tensor([[0,0,1,0]]),record_amount=torch.tensor([[.1,0,0,0]]))
        expected=rollout(model,data,.0005)[0]
        whole=account_rollout(model,data,.0005)
        self.assertLess(float((expected-whole[0]).abs().max()),1e-12)
        for cut in (1,2,3):
            prefix=account_rollout(model,data,.0005,stop=cut)
            tail=account_rollout(model,data,.0005,initial=prefix[2])
            self.assertLess(float((whole[0]-tail[0]).abs().max()),1e-12)

    def test_one_day_delay_has_no_initial_etf_trade(self):
        model=make_policy('wide')
        for p in model.parameters():
            p.data.zero_()
        data=dict(f=torch.zeros(1,2,8),spot=torch.tensor([[2.,3.,4.]],dtype=torch.float64),
                  unit=torch.tensor([[10.,20.,20.]],dtype=torch.float64),premium=torch.tensor([0.]),
                  strike_end=torch.tensor([100.]),active=torch.ones(1,2,dtype=torch.bool))
        pnl,ratio,state=account_rollout(model,data,.001,delay=True)
        self.assertEqual(float(state['pending'].detach()[0]),0.)
        self.assertEqual(float(ratio[0,0]),0.)
        self.assertEqual(float(ratio[0,1]),.25) # five absolute shares, no unit rescaling
        self.assertAlmostEqual(float(pnl),(5*(4-3)-.001*5*(3+4))/20)

if __name__=='__main__':
    unittest.main()
