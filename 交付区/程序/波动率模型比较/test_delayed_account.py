import unittest
import torch
from common_policy import CommonPolicy
from delayed_account import delayed_rollout

class DelayedAccountTests(unittest.TestCase):
    def test_first_day_zero_and_expiry_cancels_pending_order(self):
        m=CommonPolicy(False)
        for p in m.parameters():p.data.zero_()
        d=dict(f=torch.zeros(1,2,7),spot=torch.tensor([[1.,1.1,1.05]],dtype=torch.float64),
            unit=torch.ones(1,3,dtype=torch.float64),active=torch.ones(1,2,dtype=torch.bool),
            premium=torch.tensor([.1],dtype=torch.float64),strike_end=torch.tensor([1.],dtype=torch.float64))
        p,r=delayed_rollout(m,d,.001)
        self.assertAlmostEqual(float(p[0].detach()),.1-.025-.05-.00055-.000525,places=12)
        torch.testing.assert_close(r,torch.tensor([[0.,.5]],dtype=torch.float64))

if __name__=='__main__':unittest.main()
