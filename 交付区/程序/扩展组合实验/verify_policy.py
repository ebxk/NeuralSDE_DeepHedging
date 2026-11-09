"""Independent ledger, gradient and compatibility checks for extended policies."""
import importlib.util
from pathlib import Path
import unittest

import torch
from torch import nn

from extended_policy import make_policy, load_legacy, rollout


def batch(spot, unit=None, active=None):
    spot = torch.tensor(spot, dtype=torch.float64)
    n, h = spot.shape
    return dict(spot=spot, unit=torch.full_like(spot, 10.) if unit is None else
                torch.tensor(unit, dtype=torch.float64),
                f=torch.zeros(n, h-1, 8),
                active=torch.ones(n, h-1, dtype=torch.bool) if active is None else
                torch.tensor(active, dtype=torch.bool),
                premium=torch.full((n,), 5., dtype=torch.float64),
                strike_end=torch.full((n,), 100., dtype=torch.float64))


def constant(ratio=.5):
    model = make_policy("wide")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.network[-2].bias.fill_(torch.logit(torch.tensor(ratio)).item())
    return model


class PolicyChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        torch.set_num_threads(1)

    def test_entry_rebalance_and_exit_cost(self):
        data = batch([[100, 110, 90]])
        pnl, ratios, gate = rollout(constant(), data, .001)
        self.assertAlmostEqual(pnl.item(), -.95/1000, places=12)
        torch.testing.assert_close(ratios, torch.full_like(ratios, .5))
        self.assertIsNone(gate)

    def test_dividend_entitlement_adjustment_and_expiry_payment(self):
        data = batch([[100, 98, 104, 102]], [[10, 11, 11, 11]])
        data.update(strike_end=torch.tensor([99.], dtype=torch.float64),
                    record_event=torch.tensor([[7, 0, 0, 0]]),
                    ex_event=torch.tensor([[0, 7, 0, 0]]),
                    pay_event=torch.tensor([[0, 0, 0, 7]]),
                    record_amount=torch.tensor([[2., 0, 0, 0]], dtype=torch.float64))
        pnl, _, _ = rollout(constant(), data, .001)
        self.assertAlmostEqual(pnl.item(), 37.89/1000, places=12)
        data["pay_event"].zero_()
        unpaid, _, _ = rollout(constant(), data, .001)
        torch.testing.assert_close(unpaid, pnl, rtol=0, atol=1e-14)

    def test_padding_settles_once_and_does_not_recharge(self):
        data = batch([[100, 110, 110, 110], [100, 110, 90, 100]],
                     active=[[True, False, False], [True, True, True]])
        pnl, ratios, _ = rollout(constant(), data, .001)
        self.assertAlmostEqual(pnl[0].item(), -1.05/1000, places=12)
        self.assertEqual(ratios[0, 1:].abs().sum().item(), 0)

    def test_student_wealth_and_position_keep_cross_time_gradient(self):
        class Probe(nn.Module):
            def __init__(self):
                super().__init__()
                self.theta = nn.Parameter(torch.tensor(-.4, dtype=torch.float64))
            def prepare(self, data):
                return data["f"].transpose(0, 1)[..., 0]
            def action(self, prepared, previous, wealth):
                return torch.sigmoid(self.theta + 10*wealth + .2*previous), None
        data = batch([[100, 110, 100]])
        model = Probe()
        actual, _, _ = rollout(model, data, 0)
        gradient = torch.autograd.grad(actual.sum(), model.theta)[0]
        a = torch.sigmoid(model.theta + .5)
        b = torch.sigmoid(model.theta + 10*(.05 + .1*a) + .2*a)
        expected = .05 + .1*a - .1*b
        reference = torch.autograd.grad(expected, model.theta)[0]
        torch.testing.assert_close(actual[0], expected, rtol=0, atol=1e-13)
        torch.testing.assert_close(gradient, reference, rtol=0, atol=1e-13)
        center = model.theta.item()
        with torch.no_grad():
            model.theta.fill_(center + 1e-5)
            plus = rollout(model, data, 0)[0].item()
            model.theta.fill_(center - 1e-5)
            minus = rollout(model, data, 0)[0].item()
        self.assertAlmostEqual(gradient.item(), (plus-minus)/2e-5, places=9)

    def test_static_gate_endpoints_share_the_actual_account(self):
        data = batch([[100, 110, 90]], [[10, 11, 11]])
        data["f"].normal_()
        experts = dict(base=make_policy("base", wealth=True),
                       factor=make_policy("wide", wealth=True))
        for weight, name in [(0., "base"), (1., "factor")]:
            mixed = make_policy("static", wealth=True, experts=experts, fixed=weight)
            actual, _, gate = rollout(mixed, data, .0005, capture=True)
            expected = rollout(experts[name], data, .0005)[0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=1e-12)
            torch.testing.assert_close(gate, torch.full_like(gate, weight))
        gate = make_policy("reliable_gate", wealth=True, experts=experts)
        rollout(gate, data, .0005)[0].square().sum().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0
                            for p in gate.gate.parameters()))
        self.assertTrue(all(p.grad is None for p in gate.base.parameters()))

    def test_history_models_are_causal_and_capacity_matched(self):
        data = batch([[100, 110, 90, 105]])
        data["history"] = torch.randn(1, 3, 10, 4)
        data["history_valid"] = torch.ones(1, 3, 10, 4, dtype=torch.bool)
        target = sum(p.numel() for p in make_policy("wide", wealth=True, history=True).parameters())
        for kind in ["wide", "flat", "gru"]:
            model = make_policy(kind, wealth=True, history=True)
            size = sum(p.numel() for p in model.parameters())
            self.assertLessEqual(abs(size-target)/target, .05)
            original = rollout(model, data, .0005)[1]
            changed = dict(data, history=data["history"].clone())
            changed["history"][:, 2] += 100
            updated = rollout(model, changed, .0005)[1]
            torch.testing.assert_close(original[:, :2], updated[:, :2], rtol=0, atol=0)

    def test_main_networks_keep_fixed_slots_when_features_are_disabled(self):
        for kind in ["wide", "narrow", "base"]:
            shapes = []
            for wealth, history in [(False, False), (True, False), (False, True), (True, True)]:
                shapes.append(tuple((key, tuple(value.shape)) for key, value in
                                    make_policy(kind, wealth=wealth, history=history).state_dict().items()))
            self.assertTrue(all(shape == shapes[0] for shape in shapes))

    def test_residual_zero_head_reproduces_clipped_teacher(self):
        data = batch([[100, 101, 102, 103]])
        data["delta"] = torch.tensor([[0., .3, 1.]])
        data["delta_valid"] = torch.ones(1, 3, dtype=torch.bool)
        _, ratios, _ = rollout(make_policy("delta_residual"), data, 0)
        torch.testing.assert_close(ratios, data["delta"].clamp(1e-4, 1-1e-4), rtol=0, atol=1e-7)

    def test_old_weight_mapping_and_decisions(self):
        file = Path(__file__).parents[1]/"创新组合实验"/"policy.py"
        spec = importlib.util.spec_from_file_location("legacy_policy_check", file)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        data = batch([[100, 101, 102, 103], [100, 95, 98, 100]],
                     [[10, 11, 11, 11], [10, 10, 10, 10]])
        data["f"].normal_()
        for kind, width in [("wide", 46), ("narrow", 32)]:
            original = old.MLP(8, width)
            model = make_policy(kind)
            load_legacy(model, original.state_dict())
            expected = old.rollout(original, data["f"], data["unit"][:, :-1].float())[0]
            actual = rollout(model, data, .0005)[1]
            torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
