"""Causal gain regression and validation-only global thresholds; no future-token ranks."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .data import sha
from .train import write_json

class GainRouter(nn.Module):
 def __init__(self,d):
  super().__init__();self.net=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,128),nn.SiLU(),nn.Linear(128,1))
 def forward(self,x):return self.net(x.detach().float()).squeeze(-1)


def threshold(scores,budget):
 if not 0<budget<1 or not np.isfinite(scores).all():raise ValueError('Invalid calibration')
 # Strict comparison: ties skip, so never silently impose a within-sequence quota.
 return float(np.quantile(scores,1-budget,method='higher'))


def selected_nll(scores,costs,budget=.5):
 t=threshold(scores,budget);ch=np.asarray(scores)>t
 return float(np.where(ch,costs[...,1],costs[...,0]).mean(dtype=np.float64)),t,float(ch.mean())


def fit(features,costs,valid_features,valid_costs,seed,out,device,epochs=8,batch=4096,stop_after=None,resume=False):
 out=Path(out)
 def digest(a):return hashlib.sha256(memoryview(np.ascontiguousarray(a))).hexdigest()
 recipe=dict(input_sha256=[digest(a) for a in (features,costs,valid_features,valid_costs)],seed=seed,epochs=epochs,batch=batch,lr=.001,weight_decay=.01,objective='MSE(baseNLL-refineNLL)',selection='validation NLL at calibration budget .5')
 if resume and (out/'complete.json').exists():raise ValueError('Already complete')
 if resume:
  s=torch.load(out/'fullstate.pt',map_location='cpu',weights_only=False)
  if s['recipe']!=recipe:raise ValueError('Resume inputs changed')
 elif out.exists():raise ValueError('Fresh head output required')
 else:out.mkdir(parents=True)
 torch.manual_seed(seed);head=GainRouter(features.shape[-1]).to(device);opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01);gen=torch.Generator().manual_seed(seed+1)
 x=torch.as_tensor(np.asarray(features).reshape(-1,features.shape[-1]).copy(),device=device);y=torch.as_tensor(np.asarray(costs[...,0]-costs[...,1]).reshape(-1).copy(),device=device)
 vx=torch.as_tensor(np.asarray(valid_features).reshape(-1,features.shape[-1]).copy(),device=device);vc=np.asarray(valid_costs).reshape(-1,2)
 start=0;trace=[];best=float('inf');best_state=None
 if resume:
  head.load_state_dict(s['head']);opt.load_state_dict(s['optimizer']);gen.set_state(s['generator']);start=s['epoch'];trace=s['trace'];best=s['best'];best_state=s['best_head']
 for epoch in range(start,epochs):
  head.train()
  for ix in torch.randperm(len(x),generator=gen).split(batch):
   ix=ix.to(device);opt.zero_grad(set_to_none=True);loss=(head(x[ix])-y[ix]).square().mean();loss.backward();opt.step()
  head.eval()
  with torch.no_grad():scores=np.concatenate([head(vx[i:i+batch]).cpu().numpy() for i in range(0,len(vx),batch)])
  value,t,rate=selected_nll(scores,vc);trace.append(dict(epoch=epoch+1,validation_nll_50=value,threshold=t,refine_rate=rate))
  if value<best:best=value;best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
  torch.save(dict(recipe=recipe,epoch=epoch+1,head=head.state_dict(),optimizer=opt.state_dict(),generator=gen.get_state(),trace=trace,best=best,best_head=best_state),out/'fullstate.tmp');(out/'fullstate.tmp').replace(out/'fullstate.pt')
  write_json(out/'progress.json',dict(epoch=epoch+1,best=best,trace=trace))
  if stop_after is not None and epoch+1>=stop_after:return None
 head.load_state_dict(best_state);head.eval();torch.save(best_state,out/'router.pt')
 with torch.no_grad():scores=np.concatenate([head(vx[i:i+batch]).cpu().numpy() for i in range(0,len(vx),batch)])
 calibration={str(q):dict(zip(('nll','threshold','refine_rate'),selected_nll(scores,vc,q))) for q in (.25,.5,.75)}
 write_json(out/'complete.json',dict(recipe=recipe,trace=trace,best=best,calibration=calibration,router_sha256=sha(out/'router.pt'),fullstate_sha256=sha(out/'fullstate.pt')))
 return head
