"""35-input same-information control; two 46-unit layers, 3865 parameters.

Pending dividend rights are earned from this policy's own actual close shares,
not prefilled from a teacher's future holdings. prepare resets every rollout.
"""
import torch
from torch import nn
from torch.nn import functional as F


class FullStatePolicy(nn.Module):
    wealth=True
    history=False
    kind='fullstate'
    execution_delay=False

    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(35,46),nn.Tanh(),nn.Linear(46,46),nn.Tanh(),nn.Linear(46,1),nn.Sigmoid())

    def prepare(self,data):
        dtype=self.network[0].weight.dtype
        n,h=data['f'].shape[:2]
        self._units=data.get('unit',torch.ones(n,h+1,dtype=dtype)).to(dtype)
        self._notional=data.get('initial_notional',data['spot'][:,0]*self._units[:,0]
            if 'spot' in data else torch.ones(n)).to(dtype)
        self._pending=data.get('initial_pending_asset',torch.zeros(n)).to(dtype)
        self._record_amount=data.get('record_amount',torch.zeros(n,h+1)).to(dtype)
        self._ex=data.get('ex_event',torch.zeros(n,h+1))
        x=torch.cat((data['f'][...,:8].to(dtype),20*data['return_fifo'].to(dtype),
                     torch.log1p(data['raw_ewma'].to(dtype).clamp_min(0))[...,None],
                     data['delta'].to(dtype)[...,None],data['delta_valid'].to(dtype)[...,None],
                     data['notional_scale'].to(dtype)[...,None]),-1)
        first=F.linear(x.transpose(0,1).contiguous(),self.network[0].weight[:,:32],self.network[0].bias)
        return torch.cat((first,torch.arange(h,dtype=dtype)[:,None,None].expand(h,n,1)),-1)

    def action(self,prepared,previous,wealth):
        dtype=self.network[0].weight.dtype
        t=int(prepared[0,-1]);n=len(previous)
        self._pending=self._pending[:n]
        if t>0 and not self.execution_delay:
            self._pending=self._pending+previous.to(dtype)*self._units[:n,t]*self._record_amount[:n,t-1]
        self._pending=torch.where(self._ex[:n,t]>0,torch.zeros_like(self._pending),self._pending)
        if self.execution_delay:
            self._pending=self._pending+previous.to(dtype)*self._units[:n,t]*self._record_amount[:n,t]
        h=prepared[:,:46]+(self._pending/self._notional[:n])[:,None]*self.network[0].weight[:,32]
        h=h+100*wealth.to(dtype)[:,None]*self.network[0].weight[:,-2]
        h=h+previous.to(dtype)[:,None]*self.network[0].weight[:,-1]
        action=self.network[1:](h).squeeze(-1)
        return action,None
