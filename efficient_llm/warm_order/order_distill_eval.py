"""Eight teacher/order controls, prefix-only heads, independent counterfactual checks."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import torch
from .model import Model,Config
from .order_distill import IDENTITY,SWAP,objective
from .order_distill_train import train
from .order_distill_head import fit
from .recursion_eval import check_train,collect
from .strength_refinement import check
from .train import write_json,amp
from .data import Rows,sha
from .evaluate import cluster_ids
from .token_router import precision

@torch.no_grad()
def routed(m,x,ch):
 out=torch.zeros(len(x),device=x.device)
 for k,p in enumerate((IDENTITY,SWAP)):
  ix=(ch==k).nonzero().flatten()
  if len(ix):out=out.index_copy(0,ix,m.nll(x[ix],p))
 return out


def numerical(device):
 precision();torch.manual_seed(73);c=Config(d=32,heads=4,kv_heads=2,head_dim=8,ff=64,vocab=128,length=32,prefix=4);a=Model(c);teacher=copy.deepcopy(a).eval()
 for p in teacher.parameters():p.requires_grad_(False)
 b=copy.deepcopy(a).to(device);t=copy.deepcopy(teacher).to(device);x=torch.randint(128,(3,32));la,_,_=objective(a,teacher,x,'PK2',chunk=20);la.backward()
 with amp(device):lb,_,_=objective(b,t,x.to(device),'PK2',chunk=20)
 lb.backward();ga=torch.cat([p.grad.flatten() for p in a.parameters()]);gb=torch.cat([p.grad.float().cpu().flatten() for p in b.parameters()]);err=float((ga-gb).norm()/ga.norm())
 if abs(float(la.detach()-lb.detach().cpu()))>.03 or err>.08 or any(p.grad is not None for p in t.parameters()):raise ValueError('Order-distill GPU numerical gate')
 return dict(relative_gradient_error=err,cpu_loss=float(la.detach()),gpu_loss=float(lb.detach()),teacher_frozen=True)


def run(c,out,device='cuda'):
 out=Path(out)
 if out.exists():raise ValueError('Fresh output required')
 if device.startswith('cuda') and torch.cuda.device_count()!=1:raise ValueError('One GPU required')
 check(c['hashes']);precision();torch.set_num_threads(4);out.mkdir(parents=True);write_json(out/'config.json',c);tc=c['train'];start=time.monotonic();proof=numerical(device)
 train(tc,out/'train',device,stop_after=min(40,tc['steps']-1));proof['first_checkpoint']=json.loads((out/'train/checkpoint.json').read_text());train(tc,out/'train',device,resume=True);done=check_train(out/'train')
 state=torch.load(out/'train/weights.pt',map_location='cpu',weights_only=False);m=Model(Config(**tc['model']));m.load_state_dict(state['model']);m.to(device).eval()
 for p in m.parameters():p.requires_grad_(False)
 ev=c['eval'];ids=np.load(ev['indices']);rows={k:Rows(v,tc['model']['length']) for k,v in ev['stems'].items()};paths=(IDENTITY,SWAP)
 x=rows['fit'].batch(ids['valid'][:3],device);changed=x.clone();cut=tc['model']['prefix']+2;changed[:,cut:]=(changed[:,cut:]+1)%tc['model']['vocab']
 with torch.no_grad():
  for path in paths:
   if float((m.hidden(x,path)[:,:cut]-m.hidden(changed,path)[:,:cut]).abs().max())!=0:raise ValueError('Causality gate')
  changed=x.clone();changed[:,tc['model']['prefix']:]=(changed[:,tc['model']['prefix']:]+1)%tc['model']['vocab']
  if not torch.equal(m.features(x),m.features(changed)):raise ValueError('Prefix leakage')
 proof.update(causal_max=0,prefix_independent=True)
 vc,vf=collect(m,rows['fit'],ids['valid'],paths,device,True);head=None;constant=int(vc.mean(0).argmin())
 if np.any(vc[:,1]<vc[:,0]):
  fc,ff=collect(m,rows['fit'],ids['fit'],paths,device,True);head=fit(ff,fc,vf,vc,tc['seed']+839,out/'head',device,c['router_epochs']);head.eval()
  with torch.no_grad():vs=head(torch.as_tensor(vf,device=device)).cpu().numpy()
  vch=vs.argmin(1)
 else:vs=np.zeros_like(vc);vch=np.zeros(len(vc),dtype=np.int64)
 rate=np.bincount(vch,minlength=2)/len(vc);policy=dict(constant=constant,route_probabilities=rate.tolist(),head_fit=head is not None,skip_reason=None if head is not None else 'No validation example benefits from swap')
 write_json(out/'policy.json',policy);np.savez(out/'validation.npz',costs=vc,scores=vs,choices=vch,ids=ids['valid'])
 def selected(x):
  if head is None:return m.nll(x,IDENTITY)
  ch=head(m.features(x)).argmin(1);return routed(m,x,ch)
 for domain in ('dclm','fwe'):
  dest=out/domain;dest.mkdir();costs,features=collect(m,rows[domain],ids[domain],paths,device,True)
  with torch.no_grad():scores=head(torch.as_tensor(features,device=device)).cpu().numpy() if head is not None else np.zeros_like(costs)
  ch=scores.argmin(1);random=np.random.default_rng(941+tc['seed']).choice(2,len(ch),p=rate);actual=[]
  with torch.no_grad():
   for i in range(0,len(ch),8):actual.append(selected(rows[domain].batch(ids[domain][i:i+8],device)).cpu().numpy())
  actual=np.concatenate(actual);ix=np.arange(len(ch));learned=costs[ix,ch]
  if np.max(np.abs(actual-learned))>2e-4:raise ValueError('Dispatch parity')
  np.savez(dest/'summary.npz',ids=ids[domain],clusters=cluster_ids(rows[domain],ids[domain]),costs=costs,scores=scores,choices=ch,random_choices=random,learned=learned,identity=costs[:,0],swap=costs[:,1],constant=costs[:,constant],random=costs[ix,random],oracle=costs.min(1),actual=actual)
  bx=rows[domain].batch(ids[domain][:8],device);times={}
  with torch.no_grad():
   for label,fn in [('identity',lambda:m.nll(bx,IDENTITY)),('learned',lambda:selected(bx))]:
    for _ in range(2):fn()
    if bx.is_cuda:torch.cuda.synchronize()
    samples=[]
    for _ in range(5):
     t=time.perf_counter();fn()
     if bx.is_cuda:torch.cuda.synchronize()
     samples.append(time.perf_counter()-t)
    times[label]=dict(median_seconds=float(np.median(samples)),samples=samples)
  proof[domain]=times
 check(c['hashes']);check_train(out/'train');write_json(out/'complete.json',dict(status='complete',arm=tc['arm'],seed=tc['seed'],seconds=time.monotonic()-start,train=done,proof=proof,policy=policy,files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ('weights.pt','fullstate.pt')}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda');a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.out,a.device)
