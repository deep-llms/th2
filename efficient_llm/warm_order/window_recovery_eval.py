"""Evaluation-only numerical audit/recovery; never updates parent checkpoints."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import torch
from .model import Model as Dense,Config
from .window_order import Model as Grouped,bounds
from .window_canonical import Model
from .window_order_eval import PATH,collect
from .recursion_eval import check_train,latency
from .data import Rows,sha
from .train import write_json
from .strength_refinement import check
from .token_router import precision

@torch.no_grad()
def audit(reference,fast,x,windowed):
 a=reference.hidden(x,PATH);f=fast.hidden(x,PATH);rr=[q.clone() for q in getattr(reference,'last_routes',[])];fr=[q.clone() for q in getattr(fast,'last_routes',[])];err=float((a-f).abs().max());relative=float((a-f).norm()/a.norm().clamp_min(1e-12));old_max=0.
 if err>1e-4 or relative>1e-5:raise ValueError('Reference/grouped hidden disagreement '+repr((err,relative)))
 for p,q in zip(rr,fr):
  if not torch.equal(p,q):raise ValueError('Reference/grouped route mismatch')
 cuts=sorted(set([1,reference.c.prefix,min(reference.c.length-1,getattr(reference,'window',reference.c.prefix)),reference.c.length-1]))
 for cut in cuts:
  z=x.clone();z[:,cut:]=(z[:,cut:]+7)%reference.c.vocab;b=reference.hidden(z,PATH);ff=fast.hidden(z,PATH)
  if not torch.equal(a[:,:cut],b[:,:cut]):raise ValueError('Fixed-shape reference is not exactly causal')
  old_max=max(old_max,float((f[:,:cut]-ff[:,:cut]).abs().max()));e=float((b-ff).abs().max());rel=float((b-ff).norm()/b.norm().clamp_min(1e-12));err=max(err,e);relative=max(relative,rel)
  if e>1e-4 or rel>1e-5:raise ValueError('Perturbed reference/grouped disagreement')
  if windowed:
   for depth,old in enumerate(rr):
    for j,(start,end) in enumerate(bounds(x.shape[1],reference.window,(depth*reference.stride)%reference.window)):
     if start<cut:
      if not torch.equal(old[:,j],reference.last_routes[depth][:,j]):raise ValueError('Reference route leakage')
      if not torch.equal(fr[depth][:,j],fast.last_routes[depth][:,j]):raise ValueError('Grouped route instability under future perturbation')
 return dict(canonical_causal_max=0,grouped_causal_max=old_max,reference_hidden_max=err,reference_relative_error=relative,cuts=cuts,route_parity=True,policy='Exact fixed-shape causality is primary. Grouped roundoff recorded; no relaxation of original causal gate claimed.')

@torch.no_grad()
def compare_domain(reference,fast,rows,ids,device,windowed):
 a=collect(reference,rows,ids,device,windowed);b=collect(fast,rows,ids,device,windowed);delta=np.abs(a['nll']-b['nll'])
 if float(delta.max())>2e-4:raise ValueError('Full-set reference/grouped NLL mismatch')
 if windowed:np.testing.assert_array_equal(a['routes'],b['routes'])
 a['grouped_nll']=b['nll'];a['grouped_abs_delta']=delta
 return a,dict(max_nll_delta=float(delta.max()),mean_nll_delta=float(delta.mean(dtype=np.float64)),all_routes_equal=True)


def run(c,out,device='cuda'):
 out=Path(out)
 if out.exists():raise ValueError('Fresh recovery output required')
 if device.startswith('cuda') and torch.cuda.device_count()!=1:raise ValueError('One GPU required')
 check(c['hashes']);precision();torch.set_num_threads(4);out.mkdir(parents=True);write_json(out/'config.json',c);start=time.monotonic();tc=c['train'];arm=tc['arm'];windowed=arm!='D6';parent=Path(c['parent']);done=check_train(parent/'train') if windowed else None;weights=Path(c['weights']);state=torch.load(weights,map_location='cpu',weights_only=False)
 if state['config']!=tc['model']:raise ValueError('Architecture mismatch')
 m=Model(Config(**tc['model']),arm,tc['window'],tc['stride'],tc['seed']) if windowed else Dense(Config(**tc['model']));fast=Grouped(Config(**tc['model']),arm,tc['window'],tc['stride'],tc['seed']) if windowed else Dense(Config(**tc['model']))
 for model in (m,fast):
  model.load_state_dict(state['model']);model.to(device).eval()
  for p in model.parameters():p.requires_grad_(False)
 ev=c['eval'];ids=np.load(ev['indices']);rows={k:Rows(v,tc['model']['length']) for k,v in ev['stems'].items()};proof=audit(m,fast,rows['fit'].batch(ids['valid'][:3],device),windowed);write_json(out/'reference-probe.json',proof)
 for domain in ('dclm','fwe'):
  dest=out/domain;dest.mkdir();a,parity=compare_domain(m,fast,rows[domain],ids[domain],device,windowed);np.savez(dest/'summary.npz',**a)
  proof[domain]=dict(parity=parity,reference_latency=latency(m,rows[domain].batch(ids[domain][:8],device),[PATH]),grouped_latency=latency(fast,rows[domain].batch(ids[domain][:8],device),[PATH]))
  for key in ('reference_latency','grouped_latency'):proof[domain][key]['scope']='FP32 full context including CE; reference computes all experts at fixed shape; not generation'
 check(c['hashes'])
 if windowed:check_train(parent/'train')
 write_json(out/'complete.json',dict(status='complete',arm=arm,seed=tc['seed'],seconds=time.monotonic()-start,parent=str(parent),train=done,weights_sha256=sha(weights),proof=proof,files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda');a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.out,a.device)
