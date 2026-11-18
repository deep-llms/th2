"""Controlled load-unit follow-up; preserve full-context hard forward and gradients."""
import torch
from .full_context_order import Model as Full, Reference as FullReference
from .window_order import bounds

ARMS=('HT','ST','HO','SO')
class Model(Full):
 def __init__(self,c,arm,window=128,stride=64,seed=0):
  if arm not in ARMS:raise ValueError('Unknown balance arm')
  super().__init__(c,arm[0],window,stride,seed);self.balancing=arm[1]
 def layer(self,h,x,depth):
  out,ch=super().layer(h,x,depth)
  if self.training and self.balancing=='T':
   with torch.no_grad():
    spans=bounds(h.shape[1],self.window,(depth*self.stride)%self.window)
    w=torch.tensor([z-a for a,z in spans],device=ch.device,dtype=self.loads.dtype).expand(ch.shape[0],-1)
    self.loads.sub_(torch.bincount(ch.flatten(),minlength=4))
    self.loads.add_(torch.bincount(ch.flatten(),weights=w.flatten(),minlength=4))
  return out,ch
 @torch.no_grad()
 def finish_update(self):
  if self.balancing=='T':super().finish_update()
  else:
   if self.balance.count_nonzero():raise ValueError('Disabled controller has nonzero bias')
   self.loads.zero_()
class Reference(Model):
 def layer(self,h,x,depth):return FullReference.layer(self,h,x,depth)
def construct(c,reference=False):
 from .model import Config
 return (Reference if reference else Model)(Config(**c['model']),c['arm'],c['window'],c['stride'],c['seed'])
def load_initial(c):
 s=torch.load(c['initial_weights'],map_location='cpu',weights_only=False)
 if s['config']!=c['model'] or s['arm']!='D6C' or s['seed']!=c['seed']:raise ValueError('Dense parent metadata mismatch')
 m=construct(c);m.load_dense(s['model']);return m
