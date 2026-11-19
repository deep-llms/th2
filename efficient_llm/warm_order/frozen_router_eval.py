"""Frozen-body router fits with independent causal/NLL/gradient checks."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .model import Config, Model as Dense, ORDERS
from .frozen_router import Model, load_initial
from .frozen_router_train import train
from .soft_exploration_eval import audit, collect
from .soft_exploration_train import amp, write_json
from .token_router import precision
from .recursion_eval import check_train
from .strength_refinement import check
from .data import Rows, sha


class Reference(Model):
    def gates(self,h,depth):
        logits=self.depth_logits[depth].expand(*h.shape[:2],4)
        if self.arm=='L':
            y=F.layer_norm(h.to(self.route_linear.weight.dtype),(self.c.d,),self.route_norm.weight,self.route_norm.bias,self.route_norm.eps)
            logits=logits+F.linear(y,self.route_linear.weight)
        elif self.arm=='M':
            head=self.depth_heads[depth]
            y=F.layer_norm(h.to(head[1].weight.dtype),(self.c.d,),head[0].weight,head[0].bias,head[0].eps)
            logits=logits+F.linear(F.silu(F.linear(y,head[1].weight,head[1].bias)),head[3].weight,head[3].bias)
        return logits.softmax(-1)

    def hidden_and_gates(self,x,hard=False):
        pos=torch.arange(x.shape[1],device=x.device).expand_as(x);seg=torch.zeros_like(x)
        h=self.stem(self.embed(x),pos,seg);ps=[]
        for d in range(4):
            p=self.gates(h,d);outputs=[block(h,pos,seg) for block in self.middle]
            if hard:
                choice=p.argmax(-1);value=torch.zeros_like(h)
                for k in range(4):value=torch.where((choice==k)[...,None],outputs[k],value)
                h=value
            else:h=sum(p[...,k,None]*outputs[k] for k in range(4))
            ps.append(p)
        self.last_routes=[p.detach().argmax(-1) for p in ps]
        return self.norm(self.coda(h,pos,seg)),ps


def construct(c,reference=False):
    return (Reference if reference else Model)(Config(**c['model']),c['arm'],c['canonical_probability'],c['router_width'])


def numerical(device,arm):
    precision();torch.manual_seed(824)
    cfg=Config(d=16,heads=2,kv_heads=1,head_dim=8,ff=32,vocab=37,length=12,prefix=3)
    a=Reference(cfg,arm,router_width=8)
    with torch.no_grad():
        # Nonzero heads exercise all MLP derivative paths, not only zero-output initialization.
        if arm=='L':a.route_linear.weight.normal_(std=.05)
        if arm=='M':
            for head in a.depth_heads:head[-1].weight.normal_(std=.05)
    b=Model(cfg,arm,router_width=8).to(device);b.load_state_dict(a.state_dict())
    x=torch.randint(37,(2,12));la=a.nll(x).mean();la.backward()
    with amp(device):lb=b.nll(x.to(device)).mean()
    lb.backward()
    def gradients(m):return torch.cat([p.grad.detach().float().cpu().flatten() for p in m.router_parameters()])
    ga,gb=gradients(a),gradients(b)
    if not torch.isfinite(ga).all() or not torch.isfinite(gb).all() or ga.norm()==0:raise ValueError('Invalid router gradients')
    err=float((ga-gb).norm()/ga.norm());delta=abs(float(la.detach()-lb.detach().cpu()))
    if err>.08 or delta>.03:raise ValueError('Router numerical gate failed')
    if any(p.grad is not None for m in (a,b) for p in m.parameters() if not p.requires_grad):raise ValueError('Frozen gradient found')
    return dict(relative_gradient_error=err,loss_delta=delta,body_gradients_absent=True)


@torch.no_grad()
def initial(c,device):
    m=load_initial(c).to(device).eval()
    state=torch.load(Path(c['parent'])/'weights.pt',map_location='cpu',weights_only=False)
    dense=Dense(Config(**c['model'])).to(device).eval()
    dense.load_state_dict({k:state['model'][k] for k in dense.state_dict()})
    x=Rows(c['data'],c['model']['length']).batch(np.load(c['indices'])[:3],device)
    if not torch.equal(m.hidden(x,True),dense.hidden(x,ORDERS[0])):raise ValueError('Hard initial parent identity failed')
    if not torch.equal(m.nll(x,True),dense.nll(x,ORDERS[0])):raise ValueError('Hard initial parent NLL failed')
    proof=dict(hard_parent_exact=True,soft_nll_max_delta=float((m.nll(x)-dense.nll(x,ORDERS[0])).abs().max()),parent_sha256=c['parent_hashes']['weights.pt'])
    proof.update(audit(m,x));return proof


def run(c,out,device='cuda'):
    out=Path(out)
    if out.exists():raise ValueError('Fresh job required')
    check(c['hashes']);precision();tc=c['train'];torch.set_num_threads(tc.get('threads',4))
    out.mkdir(parents=True);write_json(out/'config.json',c);start=time.monotonic()
    proof=numerical(device,tc['arm']);proof['initial']=initial(tc,device)
    write_json(out/'production-probe.json',proof)
    train(tc,out/'train',device,stop_after=min(40,tc['steps']-1))
    proof['first_checkpoint']=json.loads((out/'train/checkpoint.json').read_text())
    train(tc,out/'train',device,resume=True);done=check_train(out/'train')
    proof['resume']=json.loads((out/'train/resume.json').read_text())
    if proof['resume']['loaded_fullstate_sha256']!=proof['first_checkpoint']['sha256']:raise ValueError('Resume identity failed')
    state=torch.load(out/'train/weights.pt',map_location='cpu',weights_only=False)
    m=construct(tc).to(device).eval();m.load_state_dict(state['model'])
    parent=load_initial(tc)
    m.assert_frozen(parent.frozen_state());del parent
    proof['frozen_body_exact']=True
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
        parameters=sum(p.numel() for p in m.parameters()),trainable_parameters=done['trainable_parameters'],candidate_blocks=18,
        seconds=time.monotonic()-start,files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ('weights.pt','fullstate.pt')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda')
    a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.out,a.device)
