"""Evaluate online causal heads without fitting on final scores."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import torch
from .model import Config
from .auxiliary_order import Model,PATHS
from .auxiliary_train import train
from .recursion_eval import check_train
from .evaluate import cluster_ids
from .strength_refinement import check
from .train import write_json,amp
from .data import Rows,sha
from .token_router import precision

def numerical(device):
 precision();torch.manual_seed(73);c=Config(d=16,heads=2,kv_heads=1,head_dim=8,ff=32,vocab=37,length=12,prefix=3);a=Model(c,'A');b=copy.deepcopy(a).to(device);x=torch.randint(c.vocab,(2,c.length));losses=a.objective(x,7,5);la=sum(losses);la.backward()
 with amp(device):lb=sum(b.objective(x.to(device),7,5))
 lb.backward();ga=torch.cat([p.grad.flatten() for p in a.parameters()]);gb=torch.cat([p.grad.float().cpu().flatten() for p in b.parameters()]);err=float((ga-gb).norm()/ga.norm())
 if abs(float(la.detach()-lb.detach().cpu()))>.03 or err>.08:raise ValueError('GPU numerical gate')
 return dict(relative_gradient_error=err,cpu_loss=float(la.detach()),gpu_loss=float(lb.detach()))

@torch.no_grad()
def collect(m,rows,ids,device):
 cc=[];ss=[]
 for i in range(0,len(ids),8):
  x=rows.batch(ids[i:i+8],device);cc.append(m.costs(x).cpu().numpy());ss.append(m.scores(x).cpu().numpy())
 return np.concatenate(cc),np.concatenate(ss)

def run(c,out,device='cuda'):
 out=Path(out)
 if out.exists():raise ValueError('Fresh output required')
 if device.startswith('cuda') and torch.cuda.device_count()!=1:raise ValueError('One GPU required')
 check(c['hashes']);precision();torch.set_num_threads(c['train'].get('threads',4));out.mkdir(parents=True);write_json(out/'config.json',c);tc=c['train'];start=time.monotonic();proof=numerical(device)
 write_json(out/'production-probe.json',proof);train(tc,out/'train',device,stop_after=40);proof['first_checkpoint']=json.loads((out/'train/checkpoint.json').read_text());train(tc,out/'train',device,resume=True);done=check_train(out/'train')
 if json.loads((out/'train/resume.json').read_text())['loaded_fullstate_sha256']!=proof['first_checkpoint']['sha256']:raise ValueError('First40 resume hash mismatch')
 state=torch.load(out/'train/weights.pt',map_location='cpu',weights_only=False);m=Model(Config(**tc['model']),tc['arm']);m.load_state_dict(state['model']);m.to(device).eval()
 for p in m.parameters():p.requires_grad_(False)
 torch.save(m.head.state_dict(),out/'router.pt');ev=c['eval'];ids=np.load(ev['indices']);rows={k:Rows(v,tc['model']['length']) for k,v in ev['stems'].items()};x=rows['fit'].batch(ids['valid'][:3],device);cut=tc['model']['prefix']+2;changed=x.clone();changed[:,cut:]=(changed[:,cut:]+1)%tc['model']['vocab']
 with torch.no_grad():
  for path in PATHS:
   if float((m.body.hidden(x,path)[:,:cut]-m.body.hidden(changed,path)[:,:cut]).abs().max())!=0:raise ValueError('Causality gate')
  changed=x.clone();changed[:,tc['model']['prefix']:]=(changed[:,tc['model']['prefix']:]+1)%tc['model']['vocab']
  if not torch.equal(m.scores(x),m.scores(changed)):raise ValueError('Prefix leakage')
 proof.update(causal_max=0,prefix_independent=True);vc,vs=collect(m,rows['fit'],ids['valid'],device);vch=vs.argmin(1);rate=np.bincount(vch,minlength=2)/len(vc);constant=int(vc.mean(0).argmin());policy=dict(constant=constant,route_probabilities=rate.tolist(),head='online cost regression; no endpoint head tuning')
 write_json(out/'policy.json',policy);np.savez_compressed(out/'validation.npz',costs=vc,scores=vs,choices=vch,ids=ids['valid'])
 for domain in ('dclm','fwe'):
  dest=out/domain;dest.mkdir();costs,scores=collect(m,rows[domain],ids[domain],device);ch=scores.argmin(1);random=np.random.default_rng(941+tc['seed']).choice(2,len(ch),p=rate);actual=[]
  with torch.no_grad():
   for i in range(0,len(ch),8):actual.append(m.routed_nll(rows[domain].batch(ids[domain][i:i+8],device),torch.as_tensor(ch[i:i+8],device=device)).cpu().numpy())
  actual=np.concatenate(actual);ix=np.arange(len(ch));learned=costs[ix,ch]
  if not np.isfinite(costs).all() or np.max(np.abs(actual-learned))>2e-4:raise ValueError('Dispatch parity')
  np.savez_compressed(dest/'summary.npz',ids=ids[domain],clusters=cluster_ids(rows[domain],ids[domain]),costs=costs,scores=scores,choices=ch,random_choices=random,learned=learned,identity=costs[:,0],swap=costs[:,1],uniform=costs.mean(1),constant=costs[:,constant],random=costs[ix,random],oracle=costs.min(1),actual=actual);proof[domain]=dict(max_dispatch_delta=float(np.max(np.abs(actual-learned))),route_histogram=np.bincount(ch,minlength=2).tolist())
 check(c['hashes']);check_train(out/'train');write_json(out/'complete.json',dict(status='complete',arm=tc['arm'],seed=tc['seed'],seconds=time.monotonic()-start,train=done,proof=proof,policy=policy,files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ('weights.pt','fullstate.pt')}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda');a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.out,a.device)
