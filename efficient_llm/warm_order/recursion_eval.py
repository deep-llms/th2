"""Train six shared-depth controls, evaluate two old D6; causal prefix routing."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import torch
from .model import Model as Dense,Config,ORDERS
from .recursion_model import Model,PATHS
from .recursion_train import train
from .recursion_head import fit,choose,calibrate
from .train import write_json,amp
from .data import Rows,sha
from .evaluate import cluster_ids
from .strength_refinement import check
from .token_router import precision


def check_train(path):
 path=Path(path);done=json.loads((path/'complete.json').read_text());c=json.loads((path/'recipe.json').read_text())['config'];cp=json.loads((path/'checkpoint.json').read_text())
 if done['step']!=c['steps'] or cp['step']!=c['steps'] or done['input_tokens']!=c['steps']*c['batch']*c['model']['length']:raise ValueError('Training incomplete')
 if sha(path/'weights.pt')!=done['weights_sha256'] or sha(path/'fullstate.pt')!=done['fullstate_sha256'] or cp['sha256']!=done['fullstate_sha256']:raise ValueError('Checkpoint changed')
 check(c['expected_inputs']);return done


def numerical(device):
 precision();torch.manual_seed(732);c=Config(d=32,heads=4,kv_heads=2,head_dim=8,ff=64,vocab=128,length=32,prefix=4);a=Model(c);b=copy.deepcopy(a).to(device);x=torch.randint(128,(3,32));ch=torch.tensor([0,1,2]);la=a.routed_nll(x,ch).mean();la.backward()
 with amp(device):lb=b.routed_nll(x.to(device),ch.to(device)).mean()
 lb.backward();ga=torch.cat([p.grad.flatten() for p in a.parameters()]);gb=torch.cat([p.grad.float().cpu().flatten() for p in b.parameters()]);err=float((ga-gb).norm()/ga.norm())
 if abs(float(la.detach()-lb.detach().cpu()))>.03 or err>.08:raise ValueError('Recursion BF16 gate')
 return dict(relative_gradient_error=err,cpu_loss=float(la.detach()),gpu_loss=float(lb.detach()))


@torch.no_grad()
def collect(m,rows,ids,paths,device,features=False):
 cc=[];ff=[]
 for i in range(0,len(ids),8):
  x=rows.batch(ids[i:i+8],device);cc.append(torch.stack([m.nll(x,p) for p in paths],1).cpu().numpy())
  if features:ff.append(m.features(x).cpu().numpy())
 return np.concatenate(cc),np.concatenate(ff) if features else None


@torch.no_grad()
def latency(m,x,paths,head=None,policy=None):
 def forward():
  if head is None:return m.nll(x,paths[0])
  scores=head(m.features(x)).cpu().numpy();ch=torch.as_tensor(choose(scores,policy),device=x.device);return m.routed_nll(x,ch)
 for _ in range(2):forward()
 if x.is_cuda:torch.cuda.synchronize()
 values=[]
 for _ in range(5):
  t=time.perf_counter();forward()
  if x.is_cuda:torch.cuda.synchronize()
  values.append(time.perf_counter()-t)
 return dict(median_seconds=float(np.median(values)),repeats=values,batch=len(x),length=x.shape[1],scope='Full context including CE, prefix router and dispatch; not generation, cross-job timing exploratory')


def run(c,out,device='cuda'):
 out=Path(out)
 if out.exists():raise ValueError('Fresh job required')
 if device.startswith('cuda') and torch.cuda.device_count()!=1:raise ValueError('One GPU required')
 check(c['hashes']);precision();torch.set_num_threads(4);out.mkdir(parents=True);write_json(out/'config.json',c);start=time.monotonic();tc=c['train'];arm=tc['arm'];proof={}
 if arm!='D6':
  proof=numerical(device);torch.cuda.empty_cache();train(tc,out/'train',device,stop_after=min(40,tc['steps']-1));proof['first_checkpoint']=json.loads((out/'train/checkpoint.json').read_text());train(tc,out/'train',device,resume=True);done=check_train(out/'train');weights=out/'train/weights.pt'
 else:done=None;weights=Path(c['existing_weights'])
 state=torch.load(weights,map_location='cpu',weights_only=False);m=(Dense if arm=='D6' else Model)(Config(**tc['model']));m.load_state_dict(state['model']);m.to(device).eval()
 for p in m.parameters():p.requires_grad_(False)
 ev=c['eval'];ids=np.load(ev['indices']);rows={k:Rows(v,tc['model']['length']) for k,v in ev['stems'].items()};head=None;policy=None;rate=None;constant=None
 # Full-size production forward gate, including routes chosen from prefix only.
 x=rows['fit'].batch(ids['valid'][:3],device);changed=x.clone();cut=min(tc['model']['prefix']+2,x.shape[1]-1);changed[:,cut:]=(changed[:,cut:]+1)%tc['model']['vocab'];paths=ORDERS[:1] if arm=='D6' else PATHS if arm=='RV' else (PATHS[0 if arm=='D4' else 1],)
 with torch.no_grad():
  for p in paths:
   if float((m.hidden(x,p)[:,:cut]-m.hidden(changed,p)[:,:cut]).abs().max())!=0:raise ValueError('Production causality failure')
 proof['causal_max']=0
 if arm=='RV':
  fitc,fitf=collect(m,rows['fit'],ids['fit'],PATHS,device,True);dc,df=collect(m,rows['fit'],ids['valid'],PATHS,device,True)
  head=fit(fitf,fitc,df,dc,tc['seed']+711,out/'head',device,c['router_epochs']);head.eval()
  with torch.no_grad():scores=head(torch.as_tensor(df,device=device)).cpu().numpy()
  limit=(6-2-4*tc['model']['prefix']/tc['model']['length'])/2;policy=calibrate(scores,dc,limit);ch=choose(scores,policy);rate=np.bincount(ch,minlength=3)/len(ch);constant=int(dc.mean(0).argmin())
  np.savez(out/'validation.npz',costs=dc,scores=scores,choices=ch,ids=ids['valid']);write_json(out/'policy.json',dict(policy=policy,constant=constant,route_probabilities=rate.tolist()))
  # Selected path cannot change when any suffix token is perturbed.
  changed=x.clone();changed[:,tc['model']['prefix']:]=(changed[:,tc['model']['prefix']:]+1)%tc['model']['vocab']
  with torch.no_grad():
   if not torch.equal(m.features(x),m.features(changed)):raise ValueError('Router suffix leakage')
  proof['prefix_router_independent']=True
 for d in ('dclm','fwe'):
  dest=out/d;dest.mkdir();cs,fs=collect(m,rows[d],ids[d],paths,device,arm=='RV');arr=dict(ids=ids[d],clusters=cluster_ids(rows[d],ids[d]),costs=cs);random=None
  if arm=='RV':
   with torch.no_grad():scores=head(torch.as_tensor(fs,device=device)).cpu().numpy()
   ch=choose(scores,policy);random=np.random.default_rng(7117+tc['seed']).choice(3,len(ch),p=rate);actual=[]
   with torch.no_grad():
    for i in range(0,len(ch),8):actual.append(m.routed_nll(rows[d].batch(ids[d][i:i+8],device),torch.as_tensor(ch[i:i+8],device=device)).cpu().numpy())
   actual=np.concatenate(actual);selected=cs[np.arange(len(cs)),ch]
   if float(np.max(np.abs(actual-selected)))>2e-4:raise ValueError('Grouped dispatch mismatch')
   arr.update(scores=scores,choices=ch,random_choices=random,learned=selected,random=cs[np.arange(len(cs)),random],constant=cs[:,constant],depth2=cs[:,1],oracle=cs.min(1),actual=actual,block_work=2+2*(ch+1)+4*tc['model']['prefix']/tc['model']['length'])
  else:arr.update(dense=cs[:,0])
  np.savez(dest/'summary.npz',**arr);proof[d]=dict(latency=latency(m,rows[d].batch(ids[d][:8],device),paths,head,policy))
 check(c['hashes'])
 if arm!='D6':check_train(out/'train')
 write_json(out/'complete.json',dict(status='complete',arm=arm,seed=tc['seed'],seconds=time.monotonic()-start,parameters=sum(p.numel() for p in m.parameters()),train=done,weights_sha256=sha(weights),proof=proof,files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ('weights.pt','fullstate.pt')}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda');a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.out,a.device)
