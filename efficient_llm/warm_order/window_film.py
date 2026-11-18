"""Slot-affine branch inputs; original window-routing features remain unchanged."""
import torch
from torch import nn
from .window_order import Model as Window,bounds

class Model(Window):
 def __init__(self,c,arm='WL',window=128,stride=64,seed=0,film=True):
  super().__init__(c,arm,window,stride,seed);self.film=film;self.reference=False
  self.film_gamma=nn.Parameter(torch.zeros(4,c.d),requires_grad=film)
  self.film_beta=nn.Parameter(torch.zeros(4,c.d),requires_grad=film)

 def layer(self,h,x,depth):
  spans=bounds(h.shape[1],self.window,(depth*self.stride)%self.window);ch,gates=self.choose(h,x,spans,depth);b,t,d=h.shape;w=self.window
  pieces=[];positions=[];segments=[]
  for a,z in spans:
   n=z-a;pieces.append(torch.nn.functional.pad(h[:,a:z],(0,0,0,w-n)));positions.append(torch.arange(a,a+w,device=h.device).expand(b,-1));segments.append(torch.cat([torch.zeros(b,n,device=h.device,dtype=torch.long),torch.ones(b,w-n,device=h.device,dtype=torch.long)],1))
  packed=torch.stack(pieces,1).reshape(-1,w,d);pos=torch.stack(positions,1).reshape(-1,w);seg=torch.stack(segments,1).reshape(-1,w);choice=ch.flatten();scale=gates.flatten();output=torch.zeros_like(packed)
  if self.reference:
   z=packed*(1+self.film_gamma[depth].to(packed.dtype))+self.film_beta[depth].to(packed.dtype);delta=torch.zeros_like(packed)
   for k,block in enumerate(self.middle):
    v=block(z,pos,seg)-z;delta=delta+torch.where((choice==k)[:,None,None],v,torch.zeros_like(v))
   output=packed+scale[:,None,None].to(packed.dtype)*delta
  for k,block in enumerate(self.middle):
   if self.reference:break
   ix=(choice==k).nonzero().flatten()
   if len(ix):
    q=packed[ix];z=q*(1+self.film_gamma[depth].to(q.dtype))+self.film_beta[depth].to(q.dtype)
    v=q+scale[ix,None,None].to(q.dtype)*(block(z,pos[ix],seg[ix])-z);output=output.index_copy(0,ix,v)
  output=output.reshape(b,len(spans),w,d)
  return torch.cat([output[:,j,:z-a] for j,(a,z) in enumerate(spans)],1),ch

 def load_parent(self,state):
  if any(k.startswith('film_') for k in state):raise ValueError('Expected original window parent')
  missing,unexpected=self.load_state_dict(state,strict=False)
  if set(missing)!={'film_gamma','film_beta'} or unexpected:raise ValueError('Parent schema mismatch')
  if self.film_gamma.detach().abs().max()!=0 or self.film_beta.detach().abs().max()!=0:raise ValueError('Nonidentity initialization')

ARMS=('G0','GF','L0','LF','D6C')

def construct(c):
 from .model import Config,Model as Dense
 if c['arm'] not in ARMS:raise ValueError('Unknown continuation arm')
 if c['arm']=='D6C':return Dense(Config(**c['model']))
 parent='WG' if c['arm'].startswith('G') else 'WL'
 return Model(Config(**c['model']),parent,c['window'],c['stride'],c['seed'],film=c['arm'].endswith('F'))

def load_initial(c):
 m=construct(c);s=torch.load(c['initial_weights'],map_location='cpu',weights_only=False)
 if s['config']!=c['model']:raise ValueError('Parent architecture mismatch')
 if c['arm']=='D6C':m.load_state_dict(s['model'])
 else:
  if s['arm']!=m.arm or s['window']!=m.window or s['stride']!=m.stride or s['seed']!=m.route_seed:raise ValueError('Parent routing metadata mismatch')
  m.load_parent(s['model'])
 return m
