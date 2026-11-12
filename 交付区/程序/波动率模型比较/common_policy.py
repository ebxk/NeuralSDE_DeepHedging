"""Eight fixed slots; actual previous holdings retain their gradient."""
import torch
from torch import nn
from torch.nn import functional as F

class CommonPolicy(nn.Module):
    def __init__(self,surface):
        super().__init__();self.surface=surface;self.wealth=False
        self.network=nn.Sequential(nn.Linear(8,46),nn.Tanh(),nn.Linear(46,46),nn.Tanh(),
            nn.Linear(46,1),nn.Sigmoid())

    def prepare(self,data):
        f=data['f'].to(self.network[0].weight.dtype)
        if not self.surface:f=torch.cat([f[...,:3],torch.zeros_like(f[...,3:])],dim=-1)
        return F.linear(f.transpose(0,1),self.network[0].weight[:,:7],self.network[0].bias)

    def action(self,prepared,previous,wealth):
        h=prepared+previous.to(prepared.dtype)[:,None]*self.network[0].weight[:,7]
        for layer in list(self.network)[1:]:h=layer(h)
        return h.squeeze(-1),None
