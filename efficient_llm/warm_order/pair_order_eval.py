"""Paired-order fits with independent causal/NLL/gradient checks."""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .model import Config, Model as Dense, ORDERS
from .pair_order import Model, load_initial
from .pair_order_train import train
from .soft_exploration_eval import audit
from .evaluate import cluster_ids
from .soft_exploration_train import amp, write_json
from .token_router import precision
from .recursion_eval import check_train
from .strength_refinement import check
from .data import Rows, sha


class Reference(Model):
    def gates(self,h,pair):
        if self.arm=='C':return torch.full((*h.shape[:2],2),.5,device=h.device,dtype=self.pair_logits.dtype)
        logits=self.pair_logits[pair].expand(*h.shape[:2],2)
        if self.arm=='M':
            head=self.pair_heads[pair]
            y=F.layer_norm(h.to(head[1].weight.dtype),(self.c.d,),head[0].weight,head[0].bias,head[0].eps)
            logits=logits+F.linear(F.silu(F.linear(y,head[1].weight,head[1].bias)),head[3].weight,head[3].bias)
        return logits.softmax(-1)

    def hidden_and_gates(self,x,hard=False):
        pos=torch.arange(x.shape[1],device=x.device).expand_as(x);seg=torch.zeros_like(x)
        h=self.stem(self.embed(x),pos,seg);ps=[]
        for pair in range(2):
            p=self.gates(h,pair);a,b=self.middle[2*pair:2*pair+2]
            canonical=b(a(h,pos,seg),pos,seg)
            alternate=b(a(h,pos,seg),pos,seg) if self.arm=='C' else a(b(h,pos,seg),pos,seg)
            if hard:h=torch.where((p.argmax(-1)==0)[...,None],canonical,alternate)
            else:h=p[...,:1]*canonical+p[...,1:]*alternate
            ps.append(p)
        self.last_routes=[p.detach().argmax(-1) for p in ps]
        return self.norm(self.coda(h,pos,seg)),ps


@torch.no_grad()
def collect(m,reference,rows,ids,device):
    result=dict(ids=ids,clusters=cluster_ids(rows,ids));parity={}
    for mode in ('soft','hard'):
        hard=mode=='hard';values=[];counts=[];sums=[];entropies=[];worst=0.
        for i in range(0,len(ids),8):
            x=rows.batch(ids[i:i+8],device);n=m.nll(x,hard)
            choices=[v.clone() for v in m.last_routes]
            _,ps=m.hidden_and_gates(x,hard)
            count=torch.stack([F.one_hot(ch,2).sum(1) for ch in choices],1)
            sm=torch.stack([p.sum(1) for p in ps],1)
            entropy=torch.stack([-(p*p.clamp_min(1e-30).log()).sum(-1).sum(1) for p in ps],1)
            other=reference.nll(x,hard);e=float((other-n).abs().max());worst=max(worst,e)
            if e>2e-4:raise ValueError('Fullset independent NLL parity failed')
            for a,b in zip(choices,reference.last_routes):
                if not torch.equal(a,b):raise ValueError('Independent route parity failed')
            values.append(n.cpu().numpy());counts.append(count.cpu().numpy());sums.append(sm.cpu().numpy());entropies.append(entropy.cpu().numpy())
        result[mode]=np.concatenate(values);result[mode+'_counts']=np.concatenate(counts)
        result[mode+'_probability_sums']=np.concatenate(sums);result[mode+'_entropy_sums']=np.concatenate(entropies)
        parity[mode]=worst
    if any(not np.isfinite(v).all() for v in result.values()):raise ValueError('Nonfinite evaluation')
    return result,parity


def construct(c,reference=False):
    return (Reference if reference else Model)(Config(**c['model']),c['arm'],c['canonical_probability'],c['router_width'])


def numerical(device,arm):
    precision();torch.manual_seed(824)
    cfg=Config(d=16,heads=2,kv_heads=1,head_dim=8,ff=32,vocab=37,length=12,prefix=3)
    a=Reference(cfg,arm,router_width=8)
    with torch.no_grad():
        # Nonzero heads exercise all MLP derivative paths, not only zero-output initialization.
        if arm=='M':
            for head in a.pair_heads:head[-1].weight.normal_(std=.05)
    b=Model(cfg,arm,router_width=8).to(device);b.load_state_dict(a.state_dict())
    x=torch.randint(37,(2,12));la=a.nll(x).mean();la.backward()
    with amp(device):lb=b.nll(x.to(device)).mean()
    lb.backward()
    def gradients(m):return torch.cat([p.grad.detach().float().cpu().flatten() for p in m.parameters() if p.requires_grad])
    ga,gb=gradients(a),gradients(b)
    if not torch.isfinite(ga).all() or not torch.isfinite(gb).all() or ga.norm()==0:raise ValueError('Invalid router gradients')
    err=float((ga-gb).norm()/ga.norm());delta=abs(float(la.detach()-lb.detach().cpu()))
    if err>.08 or delta>.03:raise ValueError('Router numerical gate failed')
    return dict(relative_gradient_error=err,loss_delta=delta)


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
        parameters=sum(p.numel() for p in m.parameters()),trainable_parameters=done['trainable_parameters'],candidate_blocks=10,
        seconds=time.monotonic()-start,files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ('weights.pt','fullstate.pt')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda')
    a=p.parse_args();run(json.loads(Path(a.config).read_text()),a.out,a.device)
