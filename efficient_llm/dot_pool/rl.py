"""DOT-RL v1.2: whole-layer library, detached policy, held-out RL split.

Extends the tested DOT-pool blocks/RNG utilities; does not modify old experiments.
"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .model import Config, Block, RMSNorm
from .data import sha
from .train import rng_state, restore_rng, autocast

SPEC_SHA='9c8bc06bf1c702f40e334fd23f7b0604b99661a65381e1f12b0883bb4efbe33b'
ARMS=['L','P-rand','J-RL','WS-RL']

def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(dict(value,spec_sha256=SPEC_SHA),indent=2,allow_nan=False)+'\n');tmp.replace(path)

def make_data(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(2026)
    rel=np.stack([rng.choice(200,20,replace=False) for _ in range(2000)])
    target=rng.integers(0,2000,(2000,20))
    edges=np.stack([np.repeat(np.arange(2000),20),rel.ravel()+2000,target.ravel()],-1).astype(np.int64)
    perm=rng.permutation(40000);split=np.zeros(40000,dtype=np.int8)
    split[perm[:4000]]=1;split[perm[4000:6000]]=2
    first=np.repeat(np.arange(40000),20)
    second=target.ravel()[first]*20+np.tile(np.arange(20),40000)
    two=np.stack([edges[first,0],edges[first,1],edges[second,1],edges[second,2]],-1)
    groups=[]
    for k in range(3):
        ix=np.flatnonzero((split[first]==k)&(split[second]==k));rng.shuffle(ix);groups.append(ix)
    ii,rr,oo=groups;nrl=int(.9*len(rr));no=len(oo)//2
    selection=dict(id_dev=ii[:1000],id_test=ii[1000:2000],train_compositions=ii[2000:246800],rl_train=rr[:nrl],rl_dev=rr[nrl:],ood_dev=oo[:no],ood_test=oo[no:])
    atoms=np.column_stack([edges,np.full(40000,2200)])
    arrays={k:two[ix] for k,ix in selection.items()}
    arrays.update(atoms=atoms,ood_rl_atoms=atoms[split==1],ood_test_atoms=atoms[split==2])
    arrays['train']=np.concatenate([atoms,arrays['train_compositions']])
    assert len(arrays['train'])==284800
    for k,v in arrays.items():np.save(root/(k+'.npy'),v)
    np.savez(root/'split_indices.npz',atomic=edges,split=split,first=first,second=second,**selection)
    write(root/'manifest.json',dict(seed=2026,split_order='permutation first4000 RL, next2000 OOD-test, remaining34000 ID',counts={k:len(v) for k,v in arrays.items()},files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(root.iterdir()) if p.is_file()}))
    return root

def batch(rows,device):
    x=torch.as_tensor(np.asarray(rows).copy(),device=device,dtype=torch.long)
    pos=torch.where(x[:,-1]==2200,1,2)
    return x,pos,x[torch.arange(len(x),device=device),pos+1]

class Policy(nn.Module):
    def __init__(self,d=512,steps=8,fixed=False):
        super().__init__();self.W=nn.Parameter(torch.zeros(steps,4,d),requires_grad=not fixed)
        self.E=nn.Parameter(torch.zeros(5,4));self.b=nn.Parameter(torch.zeros(steps,4));self.fixed=fixed
    def forward(self,h,s,previous):
        with torch.autocast(device_type=h.device.type,enabled=False):
            x=h.detach().float();x=x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)
            return F.linear(x,self.W[s])+self.E[previous]+self.b[s]

class Library(nn.Module):
    def __init__(self,c=None):
        super().__init__();self.c=c or Config(stem=0,coda=0,arm='L')
        self.embed_tokens=nn.Embedding(self.c.vocab,self.c.d)
        self.layers=nn.ModuleList([Block(self.c,True) for _ in range(4)])
        self.norm=RMSNorm(self.c.d)
        for m in self.modules():
            if isinstance(m,(nn.Linear,nn.Embedding)):nn.init.normal_(m.weight,std=.02)
    def counts(self):return sum(p.numel() for p in self.parameters())-self.embed_tokens.weight.numel()
    def parameter_groups(self):
        return [dict(params=[p for p in self.parameters() if p.ndim>=2],weight_decay=.1),dict(params=[p for p in self.parameters() if p.ndim<2],weight_decay=0.)]
    def forward(self,x,answer_pos,policy=None,mode='random',paths=None,policy_grad=False):
        """Grouping is per question; the chosen block processes all positions.

        In rollouts, block/embedding forwards use no_grad, while policy log-probs
        retain the only graph. In CE, policy is no_grad and samples are discrete.
        """
        b,t=x.shape;positions=torch.arange(t,device=x.device).expand_as(x)
        segments=torch.where(x==2200,-1,0);rows=torch.arange(b,device=x.device)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not policy_grad):h=self.embed_tokens(x)
        previous=torch.zeros(b,dtype=torch.long,device=x.device);chosen=[];logs=[];kls=[];ent=[]
        for s in range(8):
            if policy is not None:
                with torch.set_grad_enabled(policy_grad):
                    logits=policy(h[rows,answer_pos],s,previous);lp=logits.log_softmax(-1);p=lp.exp()
                    a=lp.argmax(-1) if mode=='greedy' else torch.multinomial(p,1).squeeze(-1)
                    if paths is not None:a=paths[:,s]
                    logs.append(lp.gather(1,a[:,None]).squeeze(1));kls.append((p*(lp+math.log(4))).sum(-1));ent.append(-(p*lp).sum(-1))
            elif paths is not None:a=paths[:,s]
            elif mode=='loop':a=torch.full((b,),s%4,device=x.device,dtype=torch.long)
            else:a=torch.randint(4,(b,),device=x.device)
            chosen.append(a);previous=a+1
            if policy is None and paths is None and mode=='loop':
                h=self.layers[s%4](h,positions,segments)
                continue
            # Index-copy restores the original example order and preserves CE gradients.
            with torch.set_grad_enabled(torch.is_grad_enabled() and not policy_grad):
                out=torch.empty_like(h)
                for k,layer in enumerate(self.layers):
                    ix=(a==k).nonzero().flatten()
                    if len(ix):out=out.index_copy(0,ix,layer(h[ix],positions[ix],segments[ix]))
                h=out
        with torch.set_grad_enabled(torch.is_grad_enabled() and not policy_grad):
            pred=F.linear(self.norm(h[rows,answer_pos]),self.embed_tokens.weight).float()
        result=dict(logits=pred,paths=torch.stack(chosen,1))
        if logs:result.update(logp=torch.stack(logs,1),kl=torch.stack(kls,1),entropy=torch.stack(ent,1))
        return result

def advantages(rewards,paths=None):
    d=rewards-rewards.mean(1,keepdim=True)
    if paths is not None:
        identical=(paths==paths[:,:1]).all(-1).all(-1)
        d=d.masked_fill(identical[:,None],0.)
    return d/(d.std(unbiased=False)+1e-6),d.std(unbiased=False)

def policy_loss(logp,kl,rewards,paths,beta=.04):
    a,scale=advantages(rewards.detach(),paths)
    return -(a*logp.sum(-1)).mean()+beta*kl.sum(-1).mean(),scale

def rollout(m,policy,rows,device,group=4):
    x,pos,y=batch(np.repeat(rows,group,axis=0),device)
    with autocast(device):r=m(x,pos,policy,policy_grad=True)
    reward=r['logits'].detach().log_softmax(-1).gather(1,y[:,None]).squeeze(1).clamp_min(-20).reshape(-1,group)
    paths=r['paths'].reshape(-1,group,8)
    loss,scale=policy_loss(r['logp'].reshape(-1,group,8),r['kl'].reshape(-1,group,8),reward,paths)
    return loss,dict(reward=reward.mean().detach(),centered_reward_std=scale.detach(),entropy=r['entropy'].mean(0).detach(),identical=(paths==paths[:,:1]).all(-1).all(-1).float().mean().detach())

class RewardStream:
    """Without replacement per pass, with serialized independent sampling state."""
    def __init__(self,n,seed):self.n=n;self.rng=np.random.default_rng(seed);self.order=self.rng.permutation(n);self.cursor=0
    def next(self,size):
        parts=[]
        while size:
            take=min(size,self.n-self.cursor);parts.append(self.order[self.cursor:self.cursor+take]);self.cursor+=take;size-=take
            if self.cursor==self.n:self.order=self.rng.permutation(self.n);self.cursor=0
        return np.concatenate(parts)
    def state(self):return dict(n=self.n,rng=self.rng.bit_generator.state,order=self.order,cursor=self.cursor)
    def load(self,s):
        assert self.n==s['n'];self.rng.bit_generator.state=s['rng'];self.order=s['order'];self.cursor=s['cursor']

def tensor_hash(m):
    h=hashlib.sha256()
    for k,v in sorted(m.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def save(path,m,p,opt,po,step,cfg,stream,extra=None):
    path=Path(path);path.mkdir(parents=True,exist_ok=False)
    state=dict(model=m.state_dict(),policy=p.state_dict() if p else None,optimizer=opt.state_dict() if opt else None,policy_optimizer=po.state_dict() if po else None,step=step,config=cfg,rng=rng_state(),stream=stream.state(),extra=extra or {},spec_sha256=SPEC_SHA)
    torch.save(state,path/'state.pt');write(path/'complete.json',dict(step=step,sha256=sha(path/'state.pt'),bytes=(path/'state.pt').stat().st_size))

def load(path,m,p,opt,po,cfg,stream,device):
    path=Path(path);r=json.loads((path/'complete.json').read_text());assert sha(path/'state.pt')==r['sha256']
    s=torch.load(path/'state.pt',map_location=device,weights_only=False)
    assert s['spec_sha256']==SPEC_SHA
    for k in ['arm','seed','model','data_sha256','source_sha256','spec_sha256','batch','lr']:assert cfg.get(k)==s['config'].get(k),k
    m.load_state_dict(s['model'])
    if p:p.load_state_dict(s['policy'])
    if opt:opt.load_state_dict(s['optimizer'])
    if po:po.load_state_dict(s['policy_optimizer'])
    stream.load(s['stream']);restore_rng(s['rng']);return s['step'],s['extra']

def choose_budget(seconds_per_epoch):
    if not math.isfinite(seconds_per_epoch) or seconds_per_epoch<=0:raise ValueError('Invalid measurement')
    n=max(2000,min(4500,math.floor(10.5*3600/seconds_per_epoch/500)*500))
    return dict(epochs=n,switch_epoch=max(500,(n//3)//500*500),seconds_per_epoch=seconds_per_epoch,minimum_overrun=n*seconds_per_epoch>10.5*3600,predicted_hours=n*seconds_per_epoch/3600)
