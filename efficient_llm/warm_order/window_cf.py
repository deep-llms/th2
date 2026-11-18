"""Frozen, causal single-decision interventions; label-using oracle is diagnostic only."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from .window_film import Model as Film
from .window_order import prior_features,bounds
from .model import Model as Dense,Config
from .data import Rows,sha
from .evaluate import cluster_ids
from .train import write_json
from .strength_refinement import check
from .token_router import precision
from .window_recovery_eval import audit
PATH=(0,1,2,3)

class Intervention(Film):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.reference=True;self.intervention=None
 def choose(self,h,x,spans,depth):
  choices,gates=super().choose(h,x,spans,depth)
  if self.intervention is None or self.intervention[0]!=depth:return choices,gates
  if self.training:raise ValueError('Interventions are evaluation-only')
  d,j,k,mode=self.intervention
  if not 0<=j<len(spans) or mode not in ('hold','rescore'):raise ValueError('Bad intervention')
  ch=choices.clone();g=gates.clone();ch[:,j]=k
  if mode=='rescore':
   probs=self.router(prior_features(h,spans).float()+self.depth[depth]).float().softmax(-1);g[:,j]=probs[:,j].gather(-1,ch[:,j,None]).squeeze(-1)
  return ch,g

@torch.no_grad()
def enumerate_costs(m,rows,ids,device,mode,batch=8):
 actions=[(d,j,k) for d in range(4) for j,_ in enumerate(bounds(m.c.length,m.window,(d*m.stride)%m.window)) for k in range(4)]
 factual=[];routes=[];costs=[];m.eval()
 for i in range(0,len(ids),batch):
  x=rows.batch(ids[i:i+batch],device);m.intervention=None;f=m.nll(x,PATH);rr=torch.cat(m.last_routes,1).cpu().numpy();factual.append(f.cpu().numpy());routes.append(rr);cc=[]
  try:
   for d,j,k in actions:
    m.intervention=(d,j,k,mode);cc.append(m.nll(x,PATH).cpu().numpy())
  finally:m.intervention=None
  costs.append(np.stack(cc,1))
 factual=np.concatenate(factual);routes=np.concatenate(routes);costs=np.concatenate(costs)
 for a,(d,j,k) in enumerate(actions):
  column=sum(len(bounds(m.c.length,m.window,(z*m.stride)%m.window)) for z in range(d))+j;mask=routes[:,column]==k
  if not np.array_equal(costs[mask,a],factual[mask]):raise ValueError('Factual intervention identity failed')
 return dict(nll=factual,costs=costs,routes=routes.astype('u1'),actions=np.asarray(actions,dtype=np.int16),ids=ids,clusters=cluster_ids(rows,ids))

@torch.no_grad()
def run(c,out,device='cuda'):
 out=Path(out)
 if out.exists():raise ValueError('Fresh diagnostic output required')
 if device.startswith('cuda') and torch.cuda.device_count()!=1:raise ValueError('One GPU per job')
 check(c['hashes']);precision();torch.set_num_threads(4);out.mkdir(parents=True);write_json(out/'config.json',c);start=time.monotonic();state=torch.load(c['weights'],map_location='cpu',weights_only=False)
 expected='GF' if c['arm']=='WG' else 'LF'
 if state['arm']!=expected or state['seed']!=c['seed']:raise ValueError('Parent endpoint mismatch')
 m=Intervention(Config(**state['config']),c['arm'],state['window'],state['stride'],c['seed'],film=True).to(device).eval();m.load_state_dict(state['model']);fast=Film(m.c,c['arm'],m.window,m.stride,c['seed'],film=True).to(device).eval();fast.load_state_dict(state['model']);fast.reference=False
 ids=np.load(c['indices'])['valid'][:c['sequences']]
 if len(ids)!=c['sequences'] or len(np.unique(ids))!=len(ids):raise ValueError('Validation reservation size')
 rows=Rows(c['data'],m.c.length);probe=audit(m,fast,rows.batch(ids[:3],device),True);del fast;torch.cuda.empty_cache();write_json(out/'probe.json',probe)
 a=enumerate_costs(m,rows,ids,device,c['mode']);del m;torch.cuda.empty_cache()
 ds=torch.load(c['dense_weights'],map_location='cpu',weights_only=False)
 if ds['arm']!='D6C' or ds['seed']!=c['seed'] or ds['config']!=state['config']:raise ValueError('Dense control mismatch')
 dense=Dense(Config(**ds['config'])).to(device).eval();dense.load_state_dict(ds['model']);a['dense_nll']=np.concatenate([dense.nll(rows.batch(ids[i:i+8],device),PATH).cpu().numpy() for i in range(0,len(ids),8)])
 np.savez(out/'costs.npz',**a);check(c['hashes']);write_json(out/'complete.json',dict(status='complete',seconds=time.monotonic()-start,identity_exact=True,probe=probe,weights_sha256=sha(c['weights']),dense_sha256=sha(c['dense_weights']),files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda');a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.out,a.device)
