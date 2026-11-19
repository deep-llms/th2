"""Detached full-information cost heads with controlled body feedback."""
import torch
from torch import nn
from .model import Model as Body
PATHS=((0,1,2,3),(0,1,3,2))

class CostHead(nn.Module):
 def __init__(self,d,global_only=False):
  super().__init__();self.global_only=global_only
  if global_only:self.scores=nn.Parameter(torch.zeros(2))
  else:
   self.net=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,128),nn.SiLU(),nn.Linear(128,2));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
 def forward(self,x):return self.scores.expand(len(x),2) if self.global_only else self.net(x.detach().float())

class Model(nn.Module):
 def __init__(self,c,arm):
  super().__init__()
  if arm not in ('D','U','G','A'):raise ValueError('Unknown arm')
  self.c=c;self.arm=arm;self.body=Body(c);self.head=CostHead(c.d,arm=='G')
 def costs(self,x):return torch.stack([self.body.nll(x,p) for p in PATHS],1)
 def scores(self,x):return self.head(self.body.features(x))
 def weights(self,scores,step,warm):
  if self.arm=='D':return torch.tensor([1.,0.],device=scores.device).expand_as(scores)
  if self.arm=='U' or step<warm:return torch.full_like(scores,.5)
  return .05+.9*(-scores.detach()/.02).softmax(-1)
 def objective(self,x,step,warm):
  costs=self.costs(x);scores=self.scores(x);target=costs.detach()-costs.detach().mean(-1,keepdim=True)
  body=(costs*self.weights(scores,step,warm)).sum(1).mean();aux=(scores-target).square().mean()
  return body,aux
 def routed_nll(self,x,choices):
  result=torch.zeros(len(x),device=x.device)
  if choices.shape!=(len(x),) or ((choices<0)|(choices>1)).any():raise ValueError('Invalid routes')
  for i,p in enumerate(PATHS):
   ix=(choices==i).nonzero().flatten()
   if len(ix):result=result.index_copy(0,ix,self.body.nll(x[ix],p))
  return result
