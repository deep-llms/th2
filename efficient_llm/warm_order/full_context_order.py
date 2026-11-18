"""Unadmitted full-prefix whole-block routing candidate; no sparse-speed claim."""
import torch
from torch import nn
from .window_order import Model as Window,bounds,prior_features


def unit_residual(p,choice,deltas,mode,outputs=None):
    """p[B,T,K], choice[B,T], deltas[B,T,K,D]; hard forward, biased route derivative."""
    if mode not in ('F','H','S') or p.shape!=deltas.shape[:-1] or choice.shape!=p.shape[:-1]:raise ValueError('Invalid route tensors')
    mask=torch.nn.functional.one_hot(choice,p.shape[-1]).bool()
    hard=deltas if outputs is None else outputs
    if hard.shape!=deltas.shape:raise ValueError('Invalid hard output shape')
    value=torch.where(mask[...,None],hard,torch.zeros_like(hard)).sum(-2)
    if mode!='F':
        candidates=deltas.detach()
        if mode=='H':candidates=torch.where(mask[...,None],candidates,torch.zeros_like(candidates))
        value=value+((p-p.detach())[...,None]*candidates).sum(-2).to(value.dtype)
    return value


class Model(Window):
    def __init__(self,c,mode='S',window=128,stride=64,seed=0):
        if mode not in ('F','H','S'):raise ValueError('Unknown full-context mode')
        super().__init__(c,'WF' if mode=='F' else 'WL',window,stride,seed)
        self.mode=mode
        self.depth_prior=nn.Parameter(torch.eye(4)*2,requires_grad=mode!='F')

    def layer(self,h,x,depth):
        spans=bounds(h.shape[1],self.window,(depth*self.stride)%self.window)
        features=prior_features(h,spans)+self.depth[depth]
        logits=self.router(features.float()).float()+self.depth_prior[depth]
        if self.training and self.mode!='F':logits=logits+.1*torch.randn_like(logits)
        p=logits.softmax(-1)
        ch=torch.full(p.shape[:-1],depth,dtype=torch.long,device=h.device) if self.mode=='F' else (p+self.balance).argmax(-1)
        if self.training and self.mode!='F':
            with torch.no_grad():self.loads.add_(torch.bincount(ch.flatten(),minlength=4))
        # Every expert sees its own full causal prefix projections of the same h.
        pos=torch.arange(h.shape[1],device=h.device).expand(h.shape[0],-1);seg=torch.zeros_like(pos)
        outputs=torch.stack([block(h,pos,seg) for block in self.middle],2)
        deltas=outputs-h[:,:,None,:]
        token_p=torch.cat([p[:,j:j+1].expand(-1,z-a,-1) for j,(a,z) in enumerate(spans)],1)
        token_ch=torch.cat([ch[:,j:j+1].expand(-1,z-a) for j,(a,z) in enumerate(spans)],1)
        # Select complete outputs directly to avoid cancellation in h+(block(h)-h).
        return unit_residual(token_p,token_ch,deltas,self.mode if self.training else 'F',outputs=outputs),ch

    def load_dense(self,state):
        missing,extra=self.load_state_dict(state,strict=False)
        expected={'depth','depth_prior','balance','loads','router.weight','router.bias'}
        if set(missing)!=expected or extra:raise ValueError('Dense parent schema mismatch')


class Reference(Model):
    """Independent window-slicing assembly of full-prefix expert outputs at eval."""
    def layer(self,h,x,depth):
        spans=bounds(h.shape[1],self.window,(depth*self.stride)%self.window)
        p=(self.router((prior_features(h,spans)+self.depth[depth]).float())+self.depth_prior[depth]).float().softmax(-1)
        ch=torch.full(p.shape[:-1],depth,dtype=torch.long,device=h.device) if self.mode=='F' else (p+self.balance).argmax(-1)
        pos=torch.arange(h.shape[1],device=h.device).expand(h.shape[0],-1);seg=torch.zeros_like(pos)
        outputs=[block(h,pos,seg) for block in self.middle]
        pieces=[]
        for j,(a,z) in enumerate(spans):
            bank=torch.stack([v[:,a:z] for v in outputs],1)
            pieces.append(bank[torch.arange(len(h),device=h.device),ch[:,j]])
        return torch.cat(pieces,1),ch

ARMS=('F','H','S')
def construct(c,reference=False):
    if c['arm'] not in ARMS:raise ValueError('Unknown full context arm')
    from .model import Config
    return (Reference if reference else Model)(Config(**c['model']),c['arm'],c['window'],c['stride'],c['seed'])

def load_initial(c):
    s=torch.load(c['initial_weights'],map_location='cpu',weights_only=False)
    if s['config']!=c['model'] or s['arm']!='D6C' or s['seed']!=c['seed']:raise ValueError('Dense parent metadata mismatch')
    m=construct(c);m.load_dense(s['model']);return m
