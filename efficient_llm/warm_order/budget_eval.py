"""Frozen-body budget routing with causal global thresholds and batched sparse queries."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from .data import sha,Rows
from .train import write_json
from .model import Config,Model
from .refinement import Refinement,canonical_state
from .refinement_eval import costs,probe
from .refinement_vectorized import sparse_block,physical_work
from .budget_head import fit
from .token_router import precision

BUDGETS=(.25,.5,.75)

def check_hashes(hashes):
 for p,h in hashes.items():
  if sha(p)!=h:raise ValueError('Changed input '+p)

@torch.no_grad()
def routed(m,head,x,cut):
 h=canonical_state(m.base,x);a=m.base.norm(h);scores=head(a[:,m.base.c.prefix-1:-1]);mask=torch.zeros_like(x,dtype=torch.bool);mask[:,m.base.c.prefix-1:-1]=scores>cut
 return m.base.norm(sparse_block(m.extra,h,mask)),mask,scores

@torch.no_grad()
def gates(m,x):
 h=canonical_state(m.base,x);mask=torch.arange(x.shape[1],device=x.device)[None,:].expand_as(x)%3==0
 pos=torch.arange(x.shape[1],device=x.device).expand_as(x);dense=m.extra(h,pos,torch.zeros_like(x));z=sparse_block(m.extra,h,mask)
 err=float((z-torch.where(mask[:,:,None],dense,h)).abs().max())
 cut=m.base.c.prefix+2;y=x.clone();y[:,cut:]=(y[:,cut:]+1)%m.base.c.vocab;hy=canonical_state(m.base,y);zy=sparse_block(m.extra,hy,mask)
 future=float((zy[:,:cut]-z[:,:cut]).abs().max())
 if err>2e-4 or future!=0:raise ValueError('Vectorized production gate failed '+str((err,future)))
 return dict(vectorized_dense_hidden_max=err,future_max=future,prior_probe=probe(m,x))

@torch.no_grad()
def benchmark(m,head,x,cut):
 def dense():
  h=canonical_state(m.base,x);pos=torch.arange(x.shape[1],device=x.device).expand_as(x)
  return m.base.norm(m.extra(h,pos,torch.zeros_like(x)))
 def sparse():return routed(m,head,x,cut)[0]
 def sync():
  if x.is_cuda:torch.cuda.synchronize()
 result={}
 # Alternating repeats; optimized dense exit omits unused base normalization.
 for include_ce in (False,True):
  timings={'dense':[],'sparse':[]}
  for k in range(7):
   for name,fun in ([('dense',dense),('sparse',sparse)] if k%2==0 else [('sparse',sparse),('dense',dense)]):
    sync();start=time.perf_counter();h=fun()
    if include_ce:costs(h,m.base.embed.weight,x,m.base.c.prefix)
    sync()
    if k>=2:timings[name].append(time.perf_counter()-start)
  result['with_ce' if include_ce else 'body']={k:float(np.median(v)) for k,v in timings.items()}
 return dict(seconds=result,batch=len(x),sequence=x.shape[1],repeats=5,scope='full context only; no decoding-speed claim')


def evaluate(c,out,device='cuda',epochs=8):
 precision();torch.set_num_threads(4);out=Path(out)
 if out.exists():raise ValueError('Fresh budget output required')
 check_hashes(c['hashes']);out.mkdir(parents=True);write_json(out/'config.json',c);start=time.monotonic()
 prior=Path(c['prior']);old=json.loads((prior/'config.json').read_text());state=torch.load(c['weights'],map_location='cpu',weights_only=False)
 m=Refinement(Model(Config(**state['config'])),frozen=True);m.load_state_dict(state['model']);m.to(device).eval()
 for p in m.parameters():p.requires_grad_(False)
 ix=np.load(old['indices']);rows={d:Rows(s,m.base.c.length) for d,s in old['stems'].items()}
 gate=gates(m,rows['fit'].batch(ix['valid'][:2],device));write_json(out/'probe.json',gate)
 head=fit(np.load(prior/'fit/features.npy',mmap_mode='r'),np.load(prior/'fit/costs.npy',mmap_mode='r'),np.load(prior/'valid/features.npy',mmap_mode='r'),np.load(prior/'valid/costs.npy',mmap_mode='r'),c['seed']+3489,out/'head',device,epochs=epochs)
 calibration=json.loads((out/'head/complete.json').read_text())['calibration'];work={}
 for domain in ('dclm','fwe'):
  dest=out/domain;dest.mkdir();previous=dict(np.load(prior/domain/'summary.npz'));ids=previous['ids'];dense_cost=np.load(prior/domain/'costs.npy',mmap_mode='r');summary={k:previous[k] for k in ('ids','clusters','base','refine','oracle')}
  baseline=np.load(c['c6'][domain]);np.testing.assert_array_equal(ids,baseline['ids']);summary['c6']=baseline['dense']
  scores=[]
  with torch.no_grad():
   for i in range(0,len(ids),8):
    x=rows[domain].batch(ids[i:i+8],device);h=m.base.norm(canonical_state(m.base,x));scores.append(head(h[:,m.base.c.prefix-1:-1]).cpu().numpy())
  scores=np.concatenate(scores);np.save(dest/'scores.npy',scores);work[domain]={}
  for budget in BUDGETS:
   key=str(budget);cut=calibration[key]['threshold'];choice=scores>cut;random=np.random.default_rng(778+c['seed']).random(scores.shape)<calibration[key]['refine_rate']
   selected=np.where(choice,dense_cost[:,:,1],dense_cost[:,:,0]);rand=np.where(random,dense_cost[:,:,1],dense_cost[:,:,0]);actual=[];error=0.;counts=dict(logical_queries=0,padded_queries=0,kv_tokens=0)
   with torch.no_grad():
    for i in range(0,len(ids),8):
     x=rows[domain].batch(ids[i:i+8],device);h,mask,score=routed(m,head,x,cut);v=costs(h,m.base.embed.weight,x,m.base.c.prefix).cpu().numpy()
     np.testing.assert_array_equal(mask[:,m.base.c.prefix-1:-1].cpu().numpy(),choice[i:i+len(x)])
     error=max(error,float(np.max(np.abs(v-selected[i:i+len(x)]))));actual.append(v.mean(1,dtype=np.float64))
     for k,vv in physical_work(mask).items():counts[k]+=vv
   if error>2e-4:raise ValueError('Sparse NLL mismatch')
   np.save(dest/f'choices-{key}.npy',choice);np.save(dest/f'random-{key}.npy',random)
   summary['learned-'+key]=selected.mean(1,dtype=np.float64);summary['random-'+key]=rand.mean(1,dtype=np.float64);summary['actual-'+key]=np.concatenate(actual);summary['count-'+key]=choice.sum(1)
   work[domain][key]=dict(**counts,router_tokens=int(scores.size),token_nll_error=error,latency=benchmark(m,head,rows[domain].batch(ids[:8],device),cut))
  np.savez(dest/'summary.npz',**summary)
 check_hashes(c['hashes'])
 write_json(out/'complete.json',dict(status='complete',name=c['name'],seconds=time.monotonic()-start,calibration=calibration,work=work,files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda');a=p.parse_args();evaluate(json.loads(Path(a.config).read_text()),a.out,a.device)
