"""Extended policies and a differentiable, self-financing contract ledger.

Historical windows contain four causal increments: return, two factor changes,
and realized-volatility change. Account wealth is scaled by 100 only at the
network boundary; all cash flows retain their original monetary units.
"""
import copy

import torch
from torch import nn
from torch.nn import functional as F


def _count(inputs, width, single=False):
    return (inputs+1)*width + width+1 + (0 if single else (width+1)*width)


def _width(inputs, target, extra=0):
    width = min(range(4, 257), key=lambda w: abs(_count(inputs, w)+extra-target))
    if abs(_count(inputs, width)+extra-target)/target > .05:
        raise ValueError("No integer hidden width matches the parameter budget")
    return width


def _history(data, dtype, base=False):
    values = data["history"].to(dtype=dtype)
    valid = data["history_valid"].bool()
    if valid.ndim == values.ndim-1:
        valid = valid[..., None].expand_as(values)
    if base:
        valid = valid.clone()
        valid[..., 1:3] = False
    return torch.where(valid, values, torch.zeros_like(values)), valid


def _summaries(values, valid):
    output = []
    for length in (5, 10):
        window, mask = values[..., -length:, :], valid[..., -length:, :]
        output.append(window.sum(dim=-2))
        variance = window[..., 0].square().sum(dim=-1)/mask[..., 0].sum(dim=-1).clamp_min(1)
        output += [variance.sqrt()[..., None], mask.to(values.dtype).mean(dim=(-1, -2))[..., None]]
    return torch.cat(output, dim=-1)


class DensePolicy(nn.Module):
    def __init__(self, kind, wealth=False, history=False, width=46,
                 current=8, single=False, target=None):
        super().__init__()
        self.kind, self.wealth, self.history = kind, wealth, history or kind in {"flat", "gru"}
        self.current = current
        extra, count = 0, current
        if kind == "gru":
            self.encoder = nn.GRU(8, 16, batch_first=True)
            extra = sum(p.numel() for p in self.encoder.parameters())
            count += 16
        elif kind == "flat":
            count += 80
        else:
            count += 12
        if kind in {"delta_input", "delta_residual"}:
            count += 2
        self.exogenous_size = count
        # Wealth/history slots stay fixed across their on/off ablations.
        inputs = count + 2
        if target is not None:
            width = _width(inputs, target, extra)
        layers = [nn.Linear(inputs, width), nn.Tanh()]
        if not single:
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1), nn.Sigmoid()]
        self.network = nn.Sequential(*layers)
        if single or kind == "delta_residual":
            nn.init.zeros_(self.network[-2].weight)
            nn.init.zeros_(self.network[-2].bias)

    def prepare(self, data):
        dtype = self.network[0].weight.dtype
        pieces = [data["f"][..., :self.current].to(dtype=dtype)]
        columns = list(range(self.current))
        if self.history:
            values, valid = _history(data, dtype, base=self.kind == "base")
            if self.kind in {"flat", "gru"}:
                frames = torch.cat([values, valid.to(dtype)], dim=-1)
                if self.kind == "flat":
                    pieces.append(frames.flatten(start_dim=-2))
                else:
                    shape = frames.shape[:2]
                    _, hidden = self.encoder(frames.reshape(-1, 10, 8))
                    pieces.append(hidden[0].reshape(*shape, 16))
                columns += list(range(self.current, self.current + (80 if self.kind == "flat" else 16)))
            else:
                pieces.append(_summaries(values, valid))
                columns += list(range(self.current, self.current+12))
        if self.kind in {"delta_input", "delta_residual"}:
            delta = data["delta"].to(dtype=dtype)
            pieces += [delta[..., None], data["delta_valid"].to(dtype)[..., None]]
            columns += [self.exogenous_size-2, self.exogenous_size-1]
        features = torch.cat(pieces, dim=-1).transpose(0, 1).contiguous()
        weights = self.network[0].weight
        weights = weights[:, :len(columns)] if columns == list(range(len(columns))) else weights[:, columns]
        result = F.linear(features, weights, self.network[0].bias)
        if self.kind == "delta_residual":
            return result, torch.logit(delta.clamp(1e-4, 1-1e-4)).transpose(0, 1)
        return result

    def action(self, prepared, previous, wealth):
        offset = None
        if self.kind == "delta_residual":
            prepared, offset = prepared
        dtype = self.network[0].weight.dtype
        h = prepared + previous.to(dtype)[:, None]*self.network[0].weight[:, -1]
        if self.wealth:
            h = h + (100*wealth.to(dtype))[:, None]*self.network[0].weight[:, -2]
        for layer in list(self.network)[1:-1]:
            h = layer(h)
        logits = h.squeeze(-1)
        return torch.sigmoid(logits if offset is None else logits+offset), None


class BlendPolicy(nn.Module):
    def __init__(self, kind, wealth, history, experts, fixed):
        super().__init__()
        if experts is None or "base" not in experts or not ({"factor", "wide"} & set(experts)):
            raise ValueError("A blend requires base and factor (or wide) experts")
        self.kind, self.fixed, self.wealth = kind, fixed, wealth
        self.base = copy.deepcopy(experts["base"]).requires_grad_(False)
        self.factor = copy.deepcopy(experts.get("factor", experts.get("wide"))).requires_grad_(False)
        self.gate = None if fixed is not None else DensePolicy(
            "gate", wealth, history, width=8, current=5 if kind == "state_gate" else 8, single=True)

    def prepare(self, data):
        return self.base.prepare(data), self.factor.prepare(data), self.gate.prepare(data) if self.gate else None

    def action(self, prepared, previous, wealth):
        base, factor, gate = prepared
        a, _ = self.base.action(base, previous, wealth)
        b, _ = self.factor.action(factor, previous, wealth)
        weight = self.gate.action(gate, previous, wealth)[0] if self.gate else torch.full_like(a, self.fixed)
        return torch.lerp(a, b, weight), weight


def make_policy(kind, wealth=False, history=False, experts=None, fixed=None):
    if kind in {"state_gate", "reliable_gate", "static"}:
        if kind == "static" and fixed is None:
            fixed = .5
        if fixed is not None and not 0 <= fixed <= 1:
            raise ValueError("A static expert weight must lie in [0,1]")
        return BlendPolicy(kind, wealth, history, experts, fixed)
    if kind not in {"wide", "narrow", "base", "capacity", "flat", "gru", "delta_input", "delta_residual"}:
        raise ValueError(f"Unknown policy {kind}")
    target = None
    if kind in {"flat", "gru"}:
        target = _count(8+12+2, 46)
    elif kind == "capacity":
        if experts is None:
            experts = dict(base=make_policy("base", wealth, history), factor=make_policy("wide", wealth, history))
        target = sum(p.numel() for m in experts.values() for p in m.parameters())
        target += _count(8+12+2, 8, single=True)
    return DensePolicy(kind, wealth, history, width=32 if kind in {"base", "narrow"} else 46,
                       current=3 if kind == "base" else 8, target=target)


def load_legacy(model, state):
    """Map original Sequential keys without changing the original checkpoint."""
    if not isinstance(model, DensePolicy) or model.wealth or model.history:
        raise ValueError("Legacy checkpoints require a plain current-state policy")
    mapped = dict(state)
    original = state["0.weight"]
    target = torch.zeros_like(model.network[0].weight)
    if original.shape[1] != model.current+1 or original.shape[0] != target.shape[0]:
        raise ValueError("Legacy input count or hidden width does not match")
    target[:, :model.current] = original[:, :-1]
    target[:, -1] = original[:, -1]
    mapped["0.weight"] = target
    model.network.load_state_dict(mapped)
    return model


def _at(prepared, index):
    if isinstance(prepared, tuple):
        return tuple(_at(value, index) for value in prepared)
    return None if prepared is None else prepared[index]


def simulated_pnl(model, data, fee):
    """Equivalent normalized wealth recursion for unit-one, no-dividend simulations."""
    spot, active = data["spot"], data["active"]
    price = spot/spot[:, :1]
    premium = data["premium"]/spot[:, 0]
    payoff = torch.relu(spot[:, -1]-data["strike_end"])/spot[:, 0]
    prepared = model.prepare(data)
    previous = torch.zeros_like(premium)
    if model.wealth:
        wealth = premium
    else:
        wealth = premium-payoff
    for t in range(active.shape[1]):
        account = wealth if model.wealth else None
        target, _ = model.action(_at(prepared, t), previous, account)
        target = torch.where(active[:, t], target, previous)
        wealth = wealth + target*(price[:, t+1]-price[:, t]) - fee*(target-previous).abs()*price[:, t]
        previous = target
    return wealth - fee*previous*price[:, -1] - (payoff if model.wealth else 0.)


def rollout(model, data, fee, capture=False):
    """Return normalized terminal wealth, active hedge ratios and gate weights.

    Floating point ledger precision follows ``spot`` (float64 for real paths).
    Event IDs are positive integers, with zero indicating no event. Each row's
    active decisions form a prefix; its first inactive day is settlement.
    """
    spot, unit, active = data["spot"], data["unit"], data["active"].bool()
    n, horizon = active.shape
    dtype, device = spot.dtype, spot.device
    if unit.shape != spot.shape or spot.shape != (n, horizon+1):
        raise ValueError("Spot and current contract units must include the terminal day")
    if horizon > 1 and (active[:, 1:] & ~active[:, :-1]).any():
        raise ValueError("Active decisions must form a prefix")
    prepared = model.prepare(data)
    initial_notional = spot[:, 0]*unit[:, 0]
    cash = data["premium"].to(dtype=dtype)*unit[:, 0]
    shares = torch.zeros(n, dtype=dtype, device=device)
    receivable = torch.zeros_like(shares)
    settled = torch.zeros(n, dtype=torch.bool, device=device)
    zeros = torch.zeros_like(active[:, 0]) if horizon else settled
    record, ex, pay = (data.get(name) for name in ("record_event", "ex_event", "pay_event"))
    event_ids = sorted(set(record[record > 0].detach().cpu().tolist())) if record is not None else []
    entitlements = {event: torch.zeros_like(shares) for event in event_ids}
    ratios, weights = [], []
    for t in range(horizon+1):
        if event_ids:
            accrued = sum(torch.where(ex[:, t] == event, amount, torch.zeros_like(amount))
                          for event, amount in entitlements.items()) if ex is not None else 0
            paid = sum(torch.where(pay[:, t] == event, amount, torch.zeros_like(amount))
                       for event, amount in entitlements.items()) if pay is not None else 0
            receivable = receivable + accrued-paid
            cash = cash + paid
        decision = active[:, t] if t < horizon else zeros
        settlement = ~decision & ~settled
        if t < horizon:
            model_dtype = next(model.parameters()).dtype
            previous = shares.to(model_dtype)/unit[:, t].to(model_dtype)
            wealth = (cash+receivable+shares*spot[:, t])/initial_notional
            ratio, weight = model.action(_at(prepared, t), previous, wealth)
            ratio = torch.where(decision, ratio, torch.zeros_like(ratio))
            ratios.append(ratio)
            if capture and weight is not None:
                weights.append(torch.where(decision, weight, torch.zeros_like(weight)))
            target = ratio.to(dtype)*unit[:, t]
        else:
            target = torch.zeros_like(shares)
        traded = target-shares
        cash = cash-traded*spot[:, t]-fee*traded.abs()*spot[:, t]
        payoff = unit[:, t]*torch.relu(spot[:, t]-data["strike_end"].to(dtype=dtype))
        cash = cash-torch.where(settlement, payoff, torch.zeros_like(payoff))
        shares = target
        for event in event_ids:
            entitlements[event] = torch.where(record[:, t] == event,
                shares*data["record_amount"][:, t].to(dtype=dtype), entitlements[event])
        settled = settled | settlement
    pnl = (cash+receivable+shares*spot[:, -1])/initial_notional
    ratio_tensor = torch.stack(ratios, dim=1) if ratios else spot[:, :0]
    return pnl, ratio_tensor, torch.stack(weights, dim=1) if weights else None
