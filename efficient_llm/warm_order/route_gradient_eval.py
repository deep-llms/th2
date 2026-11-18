"""Train causal-window controls; frozen D6 evaluation; no test-driven selection."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import torch
from .model import Model as Dense,Config
from .window_order import Model as Window,bounds
from .window_film import Model as Film
from .route_gradient import Model,construct,load_initial,residual
from dataclasses import asdict
from .route_gradient_train import train
from .window_recovery_eval import audit,compare_domain
from .recursion_eval import check_train,latency
from .train import write_json,amp
from .data import Rows,sha
from .evaluate import cluster_ids
from .strength_refinement import check
from .token_router import precision

PATH=(0,1,2,3)

def numerical(device,arm):
 precision();torch.manual_seed(732);c=Config(d=32,heads=4,kv_heads=2,head_dim=8,ff=64,vocab=128,length=31,prefix=4);a=construct(dict(model=asdict(c),arm=arm,window=8,stride=4,seed=0)).eval();b=copy.deepcopy(a).to(device);x=torch.randint(128,(3,31));la=a.nll(x,PATH).mean();la.backward()
 with amp(device):lb=b.nll(x.to(device),PATH).mean()
 lb.backward()
 def grads(m):return torch.cat([(p.grad.float() if p.grad is not None else torch.zeros_like(p)).detach().cpu().flatten() for p in m.parameters() if p.requires_grad])
 ga,gb=grads(a),grads(b);err=float((ga-gb).norm()/ga.norm());delta=abs(float(la.detach()-lb.detach().cpu()))
 if delta>.03 or err>.08:raise ValueError('Window numerical gate '+repr((delta,err)))
 for l,r in zip(getattr(a,'last_routes',[]),getattr(b,'last_routes',[])):
  if not torch.equal(l,r.cpu()):raise ValueError('CPU/GPU route mismatch')
 return dict(relative_gradient_error=err,cpu_loss=float(la.detach()),gpu_loss=float(lb.detach()),route_parity=True)

def surrogate_probe(device):
 # Analytic surrogate gradient, not finite differences of the hard objective.
 torch.manual_seed(812);z=torch.randn(5,4,device=device,requires_grad=True);p=z.softmax(-1)
 delta=torch.randn(5,4,7,8,device=device,dtype=torch.bfloat16 if device.startswith('cuda') else torch.float32,requires_grad=True)
 choice=p.argmax(-1);up=torch.randn(5,7,8,device=device)
 hard=residual(p,choice,delta,False);new=residual(p,choice,delta,True)
 if not torch.equal(hard,new):raise ValueError('Surrogate changed forward')
 gz,gd=torch.autograd.grad((new.float()*up).sum(),(z,delta))
 # Gate's cast means its backward uses the residual dtype. Extra correction is FP32.
 dot_gate=(delta.detach().float()*up[:,None]).sum((2,3)).to(delta.dtype).float()
 dot_extra=(delta.detach().float()*up[:,None].to(delta.dtype).float()).sum((2,3))
 gprob=torch.nn.functional.one_hot(choice,4)*dot_gate+p.detach()*dot_extra
 expected=p.detach()*(gprob-(p.detach()*gprob).sum(1,keepdim=True))
 err=float((gz-expected).norm()/expected.norm().clamp_min(1e-9))
 if err>.02:raise ValueError('Surrogate analytic GPU gradient mismatch '+str(err))
 mask=~torch.nn.functional.one_hot(choice,4).bool()
 if gd[mask].abs().max()!=0:raise ValueError('Unexpected unselected expert gradient')
 return dict(hard_forward_exact=True,analytic_router_relative_error=err,unselected_direct_gradient=0,estimator='biased surrogate, not finite difference or unbiased RL')

@torch.no_grad()
def causality(m,x,windowed):
 a=m.hidden(x,PATH);routes=[z.clone() for z in m.last_routes] if windowed else []
 cuts=sorted(set([1,m.c.prefix,min(m.c.length-1,getattr(m,'window',m.c.prefix)),m.c.length-1]));worst=0.
 for cut in cuts:
  z=x.clone();z[:,cut:]=(z[:,cut:]+7)%m.c.vocab;b=m.hidden(z,PATH);e=float((a[:,:cut]-b[:,:cut]).abs().max());worst=max(worst,e)
  if e>2e-5:raise ValueError('Future hidden leakage '+repr(e))
  if windowed:
   for depth,old in enumerate(routes):
    for j,(start,end) in enumerate(bounds(x.shape[1],m.window,(depth*m.stride)%m.window)):
     if start<cut and not torch.equal(old[:,j],m.last_routes[depth][:,j]):raise ValueError('Future route leakage')
 return dict(causal_max=worst,tolerance=2e-5,cuts=cuts,route_independence=True)

@torch.no_grad()
def collect(m,rows,ids,device,windowed):
 ns=[];rr=[]
 for i in range(0,len(ids),8):
  x=rows.batch(ids[i:i+8],device);ns.append(m.nll(x,PATH).cpu().numpy())
  if windowed:rr.append(torch.cat(m.last_routes,1).cpu().numpy().astype('u1'))
 a=dict(ids=ids,clusters=cluster_ids(rows,ids),nll=np.concatenate(ns))
 if windowed:
  a['routes']=np.concatenate(rr);a['route_depth']=np.asarray([d for d in range(4) for _ in bounds(m.c.length,m.window,(d*m.stride)%m.window)],dtype='u1')
 return a


@torch.no_grad()
def initial_parity(c,device):
 x=Rows(c['data'],c['model']['length']).batch(np.load(c['indices'])[:3],device)
 s=torch.load(c['initial_weights'],map_location='cpu',weights_only=False)
 parent=Dense(Config(**c['model'])) if c['arm']=='D6C' else Film(Config(**c['model']),'WG' if c['arm']=='GF' else 'WL',s['window'],s['stride'],s['seed'])
 parent.load_state_dict(s['model']);parent.reference=True;parent.to(device).eval();m=load_initial(c).to(device).eval();a=parent.hidden(x,PATH);b=m.hidden(x,PATH);delta=float((a-b).abs().max())
 if delta!=0:raise ValueError('Parent identity failure '+str(delta))
 ref=copy.deepcopy(m);ref.reference=True
 proof=audit(ref,m,x,c['arm']!='D6C');proof['identity_hidden_max']=delta
 return proof

def run(c,out,device='cuda'):
 out=Path(out)
 if out.exists():raise ValueError('Fresh job required')
 if device.startswith('cuda') and torch.cuda.device_count()!=1:raise ValueError('One GPU required')
 check(c['hashes']);precision();torch.set_num_threads(c['train'].get('threads',4));out.mkdir(parents=True);write_json(out/'config.json',c);start=time.monotonic();tc=c['train'];arm=tc['arm'];proof={};windowed=arm!='D6C'
 proof=numerical(device,arm);proof['surrogate_gradient']=surrogate_probe(device);proof['initial']=initial_parity(tc,device);torch.cuda.empty_cache();write_json(out/'production-probe.json',proof)
 train(tc,out/'train',device,stop_after=min(40,tc['steps']-1));proof['first_checkpoint']=json.loads((out/'train/checkpoint.json').read_text());train(tc,out/'train',device,resume=True);done=check_train(out/'train');weights=out/'train/weights.pt';proof['resume']=json.loads((out/'train/resume.json').read_text())
 state=torch.load(weights,map_location='cpu',weights_only=False);m=construct(tc);m.load_state_dict(state['model']);m.to(device).eval()
 for p in m.parameters():p.requires_grad_(False)
 fast=Dense(Config(**tc['model'])) if not windowed else Film(Config(**tc['model']),'WG' if arm=='GF' else 'WL',tc['window'],tc['stride'],tc['seed'])
 fast.load_state_dict(state['model']);fast.to(device).eval();m.reference=True
 for p in fast.parameters():p.requires_grad_(False)
 ev=c['eval'];ids=np.load(ev['indices']);rows={k:Rows(v,tc['model']['length']) for k,v in ev['stems'].items()};proof.update(audit(m,fast,rows['fit'].batch(ids['valid'][:3],device),windowed))
 for domain in ('dclm','fwe'):
  dest=out/domain;dest.mkdir();a,parity=compare_domain(m,fast,rows[domain],ids[domain],device,windowed)
  # Independent per-sequence forwards verify batch regrouping and route invariance.
  errors=[]
  for i in range(min(3,len(ids[domain]))):
   with torch.no_grad():n=m.nll(rows[domain].batch(ids[domain][i:i+1],device),PATH).cpu().numpy()
   errors.append(float(abs(n[0]-a['nll'][i])))
   if windowed:np.testing.assert_array_equal(torch.cat(m.last_routes,1).cpu().numpy()[0],a['routes'][i])
  if max(errors)>2e-4:raise ValueError('Batch NLL parity failure')
  np.savez(dest/'summary.npz',**a);proof[domain]=dict(latency=latency(m,rows[domain].batch(ids[domain][:8],device),[PATH]),grouped_latency=latency(fast,rows[domain].batch(ids[domain][:8],device),[PATH]),single_sequence_nll_max=max(errors),parity=parity)
  proof[domain]['latency']['scope']='Full context FP32 canonical all-expert reference including CE; not generation or sparse speed'
 check(c['hashes'])
 check_train(out/'train')
 length=tc['model']['length'];useful_attention=6*length*(length+1)//2;padded_attention=6*length*length
 if windowed:
  spans=[bounds(length,tc['window'],(d*tc['stride'])%tc['window']) for d in range(4)];useful_attention=length*(length+1)+sum((b-a)*(b-a+1)//2 for ss in spans for a,b in ss);padded_attention=length*length*2+sum(len(ss)*tc['window']**2 for ss in spans)
 proof['work']=dict(active_block_tokens_per_sequence=6*length,useful_causal_attention_pairs=useful_attention,padded_dense_attention_slots=padded_attention,canonical_expert_evaluations_per_sequence=18 if windowed else 6,scope='Active block proxy is six; window training computes four candidates per slot (18 total block evaluations). Not equal FLOPs or dense compute.')
 write_json(out/'complete.json',dict(status='complete',arm=arm,seed=tc['seed'],seconds=time.monotonic()-start,parameters=sum(p.numel() for p in m.parameters()),train=done,weights_sha256=sha(weights),proof=proof,files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ('weights.pt','fullstate.pt')}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda');a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.out,a.device)
