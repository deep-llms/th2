"""Causal shifted-window whole-block routing, with explicit fixed/gated controls."""
import torch
from torch import nn
from .model import Model as Dense

ARMS=('WF','WR','WG','WL')


def bounds(length,window,offset):
    if length<1 or window<1 or not 0<=offset<window:raise ValueError('Invalid windows')
    starts=[0]+list(range(window-offset if offset else window,length,window))
    return [(a,b) for a,b in zip(starts,starts[1:]+[length])]


def prior_features(h,spans):
    means=torch.stack([h[:,a:b].float().mean(1) for a,b in spans],1)
    prior=torch.cat([torch.zeros_like(means[:,:1]),means[:,:-1].cumsum(1)],1)
    den=torch.arange(len(spans),device=h.device).clamp_min(1)[None,:,None]
    return torch.cat([h[:,:1].float(),(prior/den)[:,1:]],1)


def random_choices(x,spans,depth,seed):
    # No future token, batch coordinate or mutable inference RNG is consulted.
    sums=x.long().cumsum(1);vals=[]
    for a,b in spans:
        z=sums[:,a-1] if a else x[:,0].long()
        z=(z+104729*(depth+1)+13007*(seed+1)+8191*a)&0x7fffffff
        z=((z^(z>>16))*73244475)&0x7fffffff
        vals.append((z^(z>>15))%4)
    return torch.stack(vals,1)


class Model(Dense):
    def __init__(self,c,arm='WL',window=128,stride=64,seed=0):
        if arm not in ARMS or window<1 or not 0<=stride<window:raise ValueError('Invalid window config')
        super().__init__(c);self.arm=arm;self.window=window;self.stride=stride;self.route_seed=seed
        self.router=nn.Linear(c.d,4);nn.init.zeros_(self.router.weight);nn.init.zeros_(self.router.bias)
        self.depth=nn.Parameter(torch.zeros(4,c.d));self.register_buffer('balance',torch.zeros(4));self.register_buffer('loads',torch.zeros(4));self.last_routes=[]
        if arm in ('WF','WR'):
            for p in self.router.parameters():p.requires_grad_(False)
            self.depth.requires_grad_(False)

    def choose(self,h,x,spans,depth):
        features=prior_features(h,spans)+self.depth[depth]
        logits=self.router(features.float()).float()
        if self.training and self.arm in ('WG','WL'):logits=logits+.1*torch.randn_like(logits)
        probs=logits.softmax(-1)
        if self.arm=='WL':ch=(probs+self.balance).argmax(-1)
        elif self.arm=='WR':ch=random_choices(x,spans,depth,self.route_seed)
        else:ch=torch.full(probs.shape[:-1],depth,device=x.device,dtype=torch.long)
        gates=probs.gather(-1,ch[...,None]).squeeze(-1) if self.arm in ('WG','WL') else torch.full_like(probs[...,0],.25)
        if self.training and self.arm=='WL':
            with torch.no_grad():self.loads.add_(torch.bincount(ch.flatten(),minlength=4))
        return ch,gates

    def layer(self,h,x,depth):
        spans=bounds(h.shape[1],self.window,(depth*self.stride)%self.window);ch,gates=self.choose(h,x,spans,depth)
        b,t,d=h.shape;w=self.window
        # Stack padded windows then group by expert; padding is a separate segment.
        pieces=[];positions=[];segments=[]
        for a,z in spans:
            n=z-a;pieces.append(torch.nn.functional.pad(h[:,a:z],(0,0,0,w-n)))
            positions.append(torch.arange(a,a+w,device=h.device).expand(b,-1))
            segments.append(torch.cat([torch.zeros(b,n,device=h.device,dtype=torch.long),torch.ones(b,w-n,device=h.device,dtype=torch.long)],1))
        packed=torch.stack(pieces,1).reshape(-1,w,d);pos=torch.stack(positions,1).reshape(-1,w);seg=torch.stack(segments,1).reshape(-1,w)
        choice=ch.flatten();scale=gates.flatten();output=torch.zeros_like(packed)
        for k,block in enumerate(self.middle):
            ix=(choice==k).nonzero().flatten()
            if len(ix):
                q=packed[ix];v=q+scale[ix,None,None].to(q.dtype)*(block(q,pos[ix],seg[ix])-q);output=output.index_copy(0,ix,v)
        output=output.reshape(b,len(spans),w,d)
        return torch.cat([output[:,j,:z-a] for j,(a,z) in enumerate(spans)],1),ch

    def hidden(self,x,path=None):
        if x.shape[1]>self.c.length:raise ValueError('Sequence too long')
        pos=torch.arange(x.shape[1],device=x.device).expand_as(x);seg=torch.zeros_like(x);h=self.stem(self.embed(x),pos,seg);routes=[]
        for depth in range(4):h,ch=self.layer(h,x,depth);routes.append(ch.detach())
        self.last_routes=routes
        return self.norm(self.coda(h,pos,seg))

    @torch.no_grad()
    def finish_update(self):
        if self.arm=='WL':self.balance.add_(.001*torch.sign(self.loads.mean()-self.loads))
        self.loads.zero_()
