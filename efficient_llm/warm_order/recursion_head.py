"""Sequence-level counterfactual cost regression and validation-only compute budget."""
import copy,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .train import write_json

class Head(nn.Module):
 def __init__(self,d):
  super().__init__();self.net=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,128),nn.SiLU(),nn.Linear(128,3))
 def forward(self,x):return self.net(x.detach().float())


def choose(scores,policy):
 if policy['force_shallow']:return np.zeros(len(scores),dtype=np.int64)
 return (scores+policy['penalty']*np.arange(1,4)).argmin(1)


def calibrate(scores,costs,max_loops):
 if not 1<=max_loops<=3:raise ValueError('Invalid compute budget')
 candidates=[dict(force_shallow=True,penalty=0.)]+[dict(force_shallow=False,penalty=float(p)) for p in np.r_[0.,np.logspace(-5,1,61)]]
 feasible=[]
 for policy in candidates:
  ch=choose(scores,policy);mean=float((ch+1).mean())
  if mean<=max_loops:feasible.append((float(costs[np.arange(len(ch)),ch].mean()),policy,mean))
 nll,policy,rate=min(feasible,key=lambda v:v[0]);return dict(**policy,validation_nll=nll,validation_mean_loops=rate,max_loops=max_loops)


def fit(f,c,df,dc,seed,out,device,epochs=30):
 out=Path(out)
 if out.exists():raise ValueError('Fresh router output required; terminal-only checkpoint')
 torch.manual_seed(seed);head=Head(f.shape[1]).to(device);opt=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
 x=torch.as_tensor(f,device=device);y=torch.as_tensor(c-c.mean(1,keepdims=True),device=device);xd=torch.as_tensor(df,device=device);yd=torch.as_tensor(dc-dc.mean(1,keepdims=True),device=device)
 gen=torch.Generator().manual_seed(seed+7031);best=float('inf');best_state=None;trace=[]
 for epoch in range(epochs):
  head.train();order=torch.randperm(len(x),generator=gen)
  for ix in order.split(256):
   ix=ix.to(device);opt.zero_grad(set_to_none=True);loss=(head(x[ix])-y[ix]).square().mean()
   if not torch.isfinite(loss):raise ValueError('Nonfinite router fit')
   loss.backward();opt.step()
  head.eval()
  with torch.no_grad():dev=float((head(xd)-yd).square().mean())
  trace.append(dict(epoch=epoch+1,validation_mse=dev))
  if dev<best:best=dev;best_state=copy.deepcopy(head.state_dict());best_epoch=epoch+1
 out.mkdir(parents=True);torch.save(dict(model=head.state_dict(),best_model=best_state,optimizer=opt.state_dict(),generator=gen.get_state(),torch_rng=torch.get_rng_state(),epochs=epochs,seed=seed,note='Terminal fullstate; no implicit resume supported'),out/'fullstate.pt');head.load_state_dict(best_state);torch.save(head.state_dict(),out/'router.pt');write_json(out/'selection.json',dict(epoch=best_epoch,dev_mse=best,trace=trace));return head
