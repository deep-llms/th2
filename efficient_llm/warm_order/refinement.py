"""Optional final refinement with exact causal sparse queries; experimental prototype."""
import copy
import torch
from torch import nn
from torch.nn import functional as F


def canonical_state(base,x):
    if x.shape[1]>base.c.length:raise ValueError('Sequence too long')
    pos=torch.arange(x.shape[1],device=x.device).expand_as(x);seg=torch.zeros_like(x)
    h=base.stem(base.embed(x),pos,seg)
    for block in base.middle:h=block(h,pos,seg)
    return base.coda(h,pos,seg)


def sparse_block(block,h,selected):
    """Exact subset of dense block outputs. K/V are from block inputs, not outputs.

    Full-context correctness path; deliberately no inference-speed claim. Only
    selected queries/FFNs are evaluated, with all input K/V for active sequences.
    """
    if selected.shape!=h.shape[:2] or selected.dtype!=torch.bool:raise ValueError('Boolean per-token mask required')
    a=block.self_attn;c=a.c;result=h.clone();t=h.shape[1]
    positions=torch.arange(t,device=h.device)
    def rotary(z,pos):
        angle=pos.float()[:,None]*a.inv;angle=torch.cat([angle,angle],-1)[:,None,:]
        return z*angle.cos().to(z.dtype)+torch.cat([-z[...,c.head_dim//2:],z[...,:c.head_dim//2]],-1)*angle.sin().to(z.dtype)
    for b in range(len(h)):
        ix=selected[b].nonzero().flatten()
        if not len(ix):continue
        x=block.input_layernorm(h[b]);q=a.q_norm(a.q_proj(x[ix]).view(len(ix),c.heads,c.head_dim))
        k=a.k_norm(a.k_proj(x).view(t,c.kv_heads,c.head_dim));v=a.v_proj(x).view(t,c.kv_heads,c.head_dim)
        q=rotary(q,ix);k=rotary(k,positions)
        mask=positions[None,:]<=ix[:,None]
        value=F.scaled_dot_product_attention(q.transpose(0,1)[None],k.transpose(0,1).repeat_interleave(c.heads//c.kv_heads,0)[None],v.transpose(0,1).repeat_interleave(c.heads//c.kv_heads,0)[None],attn_mask=mask[None,None])
        value=value[0].transpose(0,1).reshape(len(ix),-1)
        z=h[b,ix]+a.o_proj(value);z=z+block.mlp(block.post_attention_layernorm(z))
        result[b]=result[b].index_copy(0,ix,z)
    return result


class Refinement(nn.Module):
    def __init__(self,base,frozen=False):
        super().__init__();self.base=base;self.extra=copy.deepcopy(base.coda);self.frozen=frozen
        with torch.no_grad():self.extra.self_attn.o_proj.weight.zero_();self.extra.mlp.down_proj.weight.zero_()
        if frozen:
            for p in self.base.parameters():p.requires_grad_(False)
    def states(self,x,selected=None):
        if self.frozen:
            with torch.no_grad():h=canonical_state(self.base,x)
        else:h=canonical_state(self.base,x)
        if selected is None:
            pos=torch.arange(x.shape[1],device=x.device).expand_as(x)
            z=self.extra(h,pos,torch.zeros_like(x))
        else:z=sparse_block(self.extra,h,selected)
        return self.base.norm(h),self.base.norm(z)


class ExitRouter(nn.Module):
    def __init__(self,d):
        super().__init__();self.net=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,128),nn.SiLU(),nn.Linear(128,2))
    def forward(self,h):return self.net(h.detach().float())
