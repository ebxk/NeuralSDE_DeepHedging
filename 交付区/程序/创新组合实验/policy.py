"""Common hedge policy step with a cached time-independent linear term."""
import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Sequential):
    def __init__(self, size, width=32, gate=False):
        layers = [nn.Linear(size+1,width),nn.Tanh()]
        if not gate:
            layers += [nn.Linear(width,width),nn.Tanh()]
        layers += [nn.Linear(width,1),nn.Sigmoid()]
        super().__init__(*layers)
        self.size = size
        if gate:
            nn.init.zeros_(self[-2].weight)
            nn.init.zeros_(self[-2].bias)

    def prepare(self, features):
        return F.linear(features[...,:self.size],self[0].weight[:,:-1],self[0].bias)

    def step(self, features, actual_previous_ratio):
        return self.action(self.prepare(features),actual_previous_ratio)

    def action(self, prepared, previous):
        h = prepared + previous[:,None]*self[0].weight[:,-1]
        for i in range(1,len(self)):
            h = self[i](h)
        return h.squeeze(-1)


class Blend(nn.Module):
    def __init__(self, base, factor, variant, fixed=None):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.factor = factor.requires_grad_(False)
        self.variant, self.fixed = variant,fixed
        self.gate = MLP(5 if variant=="state_gate" else 8,8,gate=True) if fixed is None else None

    def prepare(self, features):
        gate = None
        if self.gate is not None:
            x = features
            if self.variant in {"residual_gate","distance_gate","constant_gate"}:
                x = x.clone()
                if self.variant!="residual_gate":
                    x[...,5] = 0
                    x[...,7] = 0
                if self.variant!="distance_gate":
                    x[...,6] = 0
            gate = self.gate.prepare(x)
        return self.base.prepare(features),self.factor.prepare(features),gate

    def step(self, features, actual_previous_ratio):
        return self.action(self.prepare(features),actual_previous_ratio)

    def weight(self, prepared, previous):
        return self.gate.action(prepared,previous) if self.gate is not None else torch.full_like(previous,self.fixed)

    def action(self, prepared, previous):
        a,b,g = prepared
        weight = self.weight(g,previous)
        return torch.lerp(self.base.action(a,previous),self.factor.action(b,previous),weight)


def prepare_steps(policy, features):
    # Transpose first to keep each decision's precomputed linear inputs contiguous.
    prepared = policy.prepare(features.transpose(0,1).contiguous())
    if isinstance(prepared,tuple):
        return zip(*(x.unbind() if x is not None else [None]*features.shape[1] for x in prepared))
    return prepared.unbind()


def rollout(policy, features, units, active=None, capture_gate=False):
    shares = torch.zeros(len(features),dtype=features.dtype)
    ratios, weights = [], []
    for t,prepared in enumerate(prepare_steps(policy,features)):
        previous = shares/units[:,t]
        ratio = policy.action(prepared,previous)
        target = ratio*units[:,t]
        shares = target if active is None else torch.where(active[:,t],target,shares)
        ratios.append(ratio)
        if capture_gate and isinstance(policy,Blend):
            weights.append(policy.weight(prepared[-1],previous))
    return torch.stack(ratios,dim=1),torch.stack(weights,dim=1) if weights else None

