"""Matched dense/global/dynamic soft fits with independent hard-projection evaluation."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import torch
from .model import Config,Model as Dense,ORDERS
from .soft_exploration import Model,load_initial
from .soft_route import ARMS
from .soft_exploration_train import train
from .recursion_eval import check_train
from .strength_refinement import check
from .train import write_json,amp
from .data import Rows,sha
from .evaluate import cluster_ids
from .token_router import precision

class Reference(Model):
    def hidden_and_gates(self,x,hard=False):
        pos=torch.arange(x.shape[1],device=x.device).expand_as(x);seg=torch.zeros_like(x)
        h=self.stem(self.embed(x),pos,seg);ps=[]
        for d in range(4):
            if self.mode=='dense':h=self.middle[d](h,pos,seg)
            else:
                p=self.gates(h,d);outputs=[b(h,pos,seg) for b in self.middle];ch=p.argmax(-1)
                if hard:
                    h=torch.zeros_like(h)
                    for k in range(4):h=torch.where((ch==k)[...,None],outputs[k],h)
                else:h=sum(p[...,k,None]*outputs[k] for k in range(4))
                ps.append(p)
        self.last_routes=[p.detach().argmax(-1) for p in ps]
        return self.norm(self.coda(h,pos,seg)),ps

def construct(c,reference=False):
    return (Reference if reference else Model)(Config(**c['model']),ARMS[c['arm']],c['canonical_probability'])

def numerical(device,mode):
    precision();torch.manual_seed(742)
    cfg=Config(d=16,heads=2,kv_heads=1,head_dim=8,ff=32,vocab=37,length=12,prefix=3)
    a=Model(cfg,mode);a.set_training_update(500);b=copy.deepcopy(a).to(device);x=torch.randint(37,(2,12))
    la=a.nll(x).mean();la.backward()
    with amp(device):lb=b.nll(x.to(device)).mean()
    lb.backward()
    def grads(m):return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).float().detach().cpu().flatten() for p in m.parameters() if p.requires_grad])
    ga,gb=grads(a),grads(b);err=float((ga-gb).norm()/ga.norm());delta=abs(float(la.detach()-lb.detach().cpu()))
    if err>.08 or delta>.03:raise ValueError('Numerical gate failed')
    return dict(relative_gradient_error=err,loss_delta=delta)

@torch.no_grad()
def audit(m,x):
    worst=0.
    for hard in (False,True):
        h,ps=m.hidden_and_gates(x,hard)
        for cut in sorted(set((1,m.c.prefix,m.c.length-1))):
            changed=x.clone();changed[:,cut:]=(changed[:,cut:]+7)%m.c.vocab
            other,qq=m.hidden_and_gates(changed,hard)
            delta=float((h[:,:cut]-other[:,:cut]).abs().max());worst=max(worst,delta)
            if delta!=0:raise ValueError('Canonical causal hidden failure')
            for a,b in zip(ps,qq):
                if not torch.equal(a[:,:cut],b[:,:cut]):raise ValueError('Causal gate failure')
    return dict(causal_max=worst,gate_causality=True)

@torch.no_grad()
def initial(c,device):
    x=Rows(c['data'],c['model']['length']).batch(np.load(c['indices'])[:3],device)
    state=torch.load(c['initial_weights'],map_location='cpu',weights_only=False)
    dense=Dense(Config(**c['model'])).to(device).eval();dense.load_state_dict(state['model'])
    m=load_initial(c).to(device).eval()
    if not torch.equal(m.hidden(x,True),dense.hidden(x,ORDERS[0])):raise ValueError('Initial hard identity failed')
    delta=float((m.nll(x)-dense.nll(x,ORDERS[0])).abs().max())
    if delta>.01:raise ValueError('Initial soft perturbation exceeds registered tolerance')
    return dict(hard_identity_exact=True,soft_nll_max_delta=delta,training_examples=3)

@torch.no_grad()
def collect(m,reference,rows,ids,device):
    values={'soft':[],'hard':[]};counts={'soft':[],'hard':[]};sums={'soft':[],'hard':[]};parity={}
    for mode in values:
        hard=mode=='hard';worst=0.
        for i in range(0,len(ids),8):
            x=rows.batch(ids[i:i+8],device);n=m.nll(x,hard)
            choices=[v.clone() for v in m.last_routes]
            # Capture probabilities independently without relying on retained autograd graphs.
            _,ps=m.hidden_and_gates(x,hard)
            if ps:
                count=torch.stack([torch.nn.functional.one_hot(ch,4).sum(1) for ch in choices],1)
                sm=torch.stack([p.sum(1) for p in ps],1)
            else:
                count=torch.eye(4,device=device,dtype=torch.long)[None].expand(len(x),4,4)*x.shape[1]
                sm=count.float()
            other=reference.nll(x,hard);e=float((other-n).abs().max());worst=max(worst,e)
            if e>2e-4:raise ValueError('Fullset independent NLL parity failed')
            for a,b in zip(choices,reference.last_routes):
                if not torch.equal(a,b):raise ValueError('Independent route parity failed')
            values[mode].append(n.cpu().numpy());counts[mode].append(count.cpu().numpy());sums[mode].append(sm.cpu().numpy())
        parity[mode]=worst
    a=dict(ids=ids,clusters=cluster_ids(rows,ids))
    for mode in values:
        a[mode]=np.concatenate(values[mode]);a[mode+'_counts']=np.concatenate(counts[mode]);a[mode+'_probability_sums']=np.concatenate(sums[mode])
    if any(not np.isfinite(v).all() for v in a.values()):raise ValueError('Nonfinite evaluation')
    return a,parity

def run(c,out,device='cuda'):
    out=Path(out)
    if out.exists():raise ValueError('Fresh job required')
    if device.startswith('cuda') and torch.cuda.device_count()!=1:raise ValueError('One GPU required')
    check(c['hashes']);precision();tc=c['train'];torch.set_num_threads(tc.get('threads',4))
    out.mkdir(parents=True);write_json(out/'config.json',c);start=time.monotonic()
    proof=numerical(device,ARMS[tc['arm']]);proof['initial']=initial(tc,device)
    probe=load_initial(tc).to(device).train();probe.set_training_update(500)
    x=Rows(tc['data'],tc['model']['length']).batch(np.load(tc['indices'])[:3],device)
    proof['floor_causality']=audit(probe,x)
    if probe.epsilon!=.4:raise ValueError('Exploration probe schedule')
    del probe,x
    write_json(out/'production-probe.json',proof)
    train(tc,out/'train',device,stop_after=min(40,tc['steps']-1));proof['first_checkpoint']=json.loads((out/'train/checkpoint.json').read_text())
    train(tc,out/'train',device,resume=True);done=check_train(out/'train')
    resume=json.loads((out/'train/resume.json').read_text())
    if resume['loaded_fullstate_sha256']!=proof['first_checkpoint']['sha256']:raise ValueError('Resume hash failure')
    proof['resume']=resume
    state=torch.load(out/'train/weights.pt',map_location='cpu',weights_only=False)
    m=construct(tc).to(device).eval();m.load_state_dict(state['model'])
    ref=construct(tc,True).to(device).eval();ref.load_state_dict(state['model'])
    for model in (m,ref):
        for p in model.parameters():p.requires_grad_(False)
    ev=c['eval'];ids=np.load(ev['indices']);rows={k:Rows(v,tc['model']['length']) for k,v in ev['stems'].items()}
    proof.update(audit(m,rows['fit'].batch(ids['valid'][:3],device)))
    for domain in ('dclm','fwe'):
        dest=out/domain;dest.mkdir();a,parity=collect(m,ref,rows[domain],ids[domain],device)
        np.savez_compressed(dest/'summary.npz',**a);proof[domain]=dict(fullset_nll_parity=parity)
    check(c['hashes']);check_train(out/'train')
    write_json(out/'complete.json',dict(status='complete',arm=tc['arm'],seed=tc['seed'],train=done,proof=proof,
        parameters=sum(p.numel() for p in m.parameters()),candidate_blocks=6 if tc['arm']=='C' else 18,seconds=time.monotonic()-start,
        files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ('weights.pt','fullstate.pt')}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda');a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.out,a.device)
