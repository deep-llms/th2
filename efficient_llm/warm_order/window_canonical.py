"""Fixed-shape reference dispatch for auditing grouped-window numerical effects."""
import torch
from .window_order import Model as Grouped,bounds

class Model(Grouped):
 def layer(self,h,x,depth):
  spans=bounds(h.shape[1],self.window,(depth*self.stride)%self.window);ch,gates=self.choose(h,x,spans,depth);b,t,d=h.shape;w=self.window
  pieces=[];positions=[];segments=[]
  for a,z in spans:
   n=z-a;pieces.append(torch.nn.functional.pad(h[:,a:z],(0,0,0,w-n)));positions.append(torch.arange(a,a+w,device=h.device).expand(b,-1));segments.append(torch.cat([torch.zeros(b,n,device=h.device,dtype=torch.long),torch.ones(b,w-n,device=h.device,dtype=torch.long)],1))
  packed=torch.stack(pieces,1).reshape(-1,w,d);pos=torch.stack(positions,1).reshape(-1,w);seg=torch.stack(segments,1).reshape(-1,w);choice=ch.flatten();scale=gates.flatten();delta=torch.zeros_like(packed)
  # All four blocks receive the same fixed shape, independent of any route.
  # This is evaluation/reference work, not a sparse-speed implementation.
  for k,block in enumerate(self.middle):
   v=block(packed,pos,seg)-packed;delta=delta+torch.where((choice==k)[:,None,None],v,torch.zeros_like(v))
  output=(packed+scale[:,None,None].to(packed.dtype)*delta).reshape(b,len(spans),w,d)
  return torch.cat([output[:,j,:z-a] for j,(a,z) in enumerate(spans)],1),ch
