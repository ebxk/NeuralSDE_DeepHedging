import unittest
import bootstrap
import numpy as np
import torch
from bottleneck_core import budget_limits, constrained_rank, costs, jobs

class CoreTests(unittest.TestCase):
    def test_rmse_budget_is_squared(self):
        mse, fee = budget_limits(.0001, .001, .02)
        self.assertAlmostEqual(mse, 1.0404)
        self.assertAlmostEqual(fee, .00105)

    def test_feasibility_precedes_tail(self):
        self.assertLess(constrained_rank(.99, .99, 3), constrained_rank(1.01, .99, 1))
        self.assertLess(constrained_rank(.99, .99, 1), constrained_rank(.99, .99, 3))

    def test_all117_registered33_unconditional(self):
        task = jobs()
        self.assertEqual(len(task), 117)
        self.assertEqual(len({j['key'] for j in task}), 117)
        self.assertEqual(sum(j['trigger'] == 'always' for j in task), 33)

    def test_fees_include_open_resize_close_with_units(self):
        data = {'spot':torch.tensor([[2.,3.,4.]],dtype=torch.float64),
                'unit':torch.tensor([[10.,20.,20.]]),
                'active':torch.tensor([[True,True]])}
        # Five shares, then ten shares, then close ten; unit change is not a free gift.
        got = costs(torch.tensor([[.5,.5]]), data, .001)
        self.assertAlmostEqual(float(got), .001*(5*2+5*3+10*4)/20)

if __name__ == '__main__':
    unittest.main()
