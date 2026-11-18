"""Experimental biased surrogate for a hard expert choice; not an RL estimator."""
import torch
import torch.nn.functional as F
from .window_film import Model as Film
from .window_order import bounds,prior_features


def residual(probabilities, choices, deltas, surrogate=False):
    """[N,K] probabilities, [N] choices, [N,K,T,D] candidate residuals.

    Preserve the hard selected-probability forward exactly. The extra derivative
    compares detached candidate residuals; unselected operators receive no direct
    gradient from this correction. Intended only for a separately controlled run.
    """
    if probabilities.ndim!=2 or deltas.ndim!=4 or deltas.shape[:2]!=probabilities.shape:
        raise ValueError('Invalid routing tensor shapes')
    mask=F.one_hot(choices,num_classes=probabilities.shape[1]).bool()
    selected=torch.where(mask[:,:,None,None],deltas,torch.zeros_like(deltas)).sum(1)
    scale=probabilities.gather(1,choices[:,None]).squeeze(1)
    output=selected*scale[:,None,None].to(selected.dtype)
    if surrogate:
        zero=probabilities-probabilities.detach()
        candidate=probabilities.detach()[:,:,None,None]*deltas.detach()
        correction=(zero[:,:,None,None]*candidate).sum(1).to(output.dtype)
        output=output+correction
    return output


class Model(Film):
    """Canonical all-expert hard forward with an optional routing surrogate."""
    def __init__(self,*args,surrogate=False,**kwargs):
        super().__init__(*args,**kwargs);self.surrogate=surrogate;self.reference=True

    def choose(self,h,x,spans,depth):
        if self.arm not in ('WG','WL'):raise ValueError('Only gated window parents supported')
        features=prior_features(h,spans)+self.depth[depth]
        logits=self.router(features.float()).float()
        if self.training:logits=logits+.1*torch.randn_like(logits)
        p=logits.softmax(-1);self.current_probs=p
        choice=(p+self.balance).argmax(-1) if self.arm=='WL' else torch.full(p.shape[:-1],depth,dtype=torch.long,device=h.device)
        if self.training and self.arm=='WL':
            with torch.no_grad():self.loads.add_(torch.bincount(choice.flatten(),minlength=4))
        return choice,p.gather(-1,choice[...,None]).squeeze(-1)

    def layer(self,h,x,depth):
        spans=bounds(h.shape[1],self.window,(depth*self.stride)%self.window)
        choice,_=self.choose(h,x,spans,depth);b,t,d=h.shape;w=self.window
        pieces=[];positions=[];segments=[]
        for a,z in spans:
            n=z-a;pieces.append(F.pad(h[:,a:z],(0,0,0,w-n)))
            positions.append(torch.arange(a,a+w,device=h.device).expand(b,-1))
            segments.append(torch.cat([torch.zeros(b,n,device=h.device,dtype=torch.long),torch.ones(b,w-n,device=h.device,dtype=torch.long)],1))
        packed=torch.stack(pieces,1).reshape(-1,w,d);pos=torch.stack(positions,1).reshape(-1,w);seg=torch.stack(segments,1).reshape(-1,w)
        z=packed*(1+self.film_gamma[depth].to(packed.dtype))+self.film_beta[depth].to(packed.dtype)
        deltas=torch.stack([block(z,pos,seg)-z for block in self.middle],1)
        output=packed+residual(self.current_probs.reshape(-1,4),choice.flatten(),deltas,self.surrogate and self.training and self.arm=='WL')
        output=output.reshape(b,len(spans),w,d)
        return torch.cat([output[:,j,:z-a] for j,(a,z) in enumerate(spans)],1),choice


ARMS=('HG','SG','GF','D6C')

def construct(c):
    from .model import Config,Model as Dense
    if c['arm'] not in ARMS:raise ValueError('Unknown gradient study arm')
    if c['arm']=='D6C':return Dense(Config(**c['model']))
    return Model(Config(**c['model']),'WG' if c['arm']=='GF' else 'WL',c['window'],c['stride'],c['seed'],film=True,surrogate=c['arm']=='SG')

def load_initial(c):
    m=construct(c);state=torch.load(c['initial_weights'],map_location='cpu',weights_only=False)
    expected='D6C' if c['arm']=='D6C' else 'GF' if c['arm']=='GF' else 'LF'
    if state['config']!=c['model'] or state['arm']!=expected or state['seed']!=c['seed']:raise ValueError('Parent metadata mismatch')
    if c['arm']!='D6C' and (state['window']!=c['window'] or state['stride']!=c['stride']):raise ValueError('Window mismatch')
    m.load_state_dict(state['model']);return m
