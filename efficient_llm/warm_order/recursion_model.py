"""Fixed/variable-depth shared middle stack; prefix-only per-sequence routing."""
import torch
from torch import nn
from .model import Config,Model as Dense
PATHS=tuple((0,1)*r for r in (1,2,3))

def path_for(arm,step,total):
 if arm=='D4':return PATHS[0]
 if arm=='RF':return PATHS[1]
 if arm=='RV':return PATHS[(0,1,1,2)[step%4]]
 raise ValueError('Unknown recursive arm')

class Model(Dense):
 def __init__(self,c):
  super().__init__(c)
  self.middle=nn.ModuleList(list(self.middle)[:2])
 def hidden(self,x,path):
  if tuple(path) not in PATHS or x.shape[1]>self.c.length:raise ValueError('Invalid recursive path or length')
  pos=torch.arange(x.shape[1],device=x.device).expand_as(x);seg=torch.zeros_like(x);h=self.stem(self.embed(x),pos,seg)
  for i in path:h=self.middle[i](h,pos,seg)
  return self.norm(self.coda(h,pos,seg))
 @torch.no_grad()
 def features(self,x):return self.hidden(x[:,:self.c.prefix],PATHS[0])[:,-1].float().detach()
 def routed_nll(self,x,choices):
  if choices.shape!=(len(x),) or choices.dtype!=torch.long or ((choices<0)|(choices>=3)).any():raise ValueError('Invalid route choices')
  result=torch.zeros(len(x),device=x.device)
  for k,path in enumerate(PATHS):
   ix=(choices==k).nonzero().flatten()
   if len(ix):result=result.index_copy(0,ix,self.nll(x[ix],path))
  return result
