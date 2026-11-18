"""Batched sparse refinement: pad selected queries, retain full causal input K/V."""
import torch
from torch.nn import functional as F


def sparse_block(block,h,selected):
    if selected.shape!=h.shape[:2] or selected.dtype!=torch.bool:raise ValueError('Boolean mask required')
    counts=selected.sum(1);active=counts.nonzero().flatten();result=h.clone()
    if not len(active):return result
    a=block.self_attn;c=a.c;t=h.shape[1];mask=selected[active];n=len(active);width=int(counts.max())
    positions=torch.arange(t,device=h.device)
    # Sort original positions, moving unselected positions after all real queries.
    ix=torch.where(mask,positions[None,:],t).sort(1).values[:,:width]
    valid=ix<t;ix=ix.clamp_max(t-1)
    x=block.input_layernorm(h[active]);query=x.gather(1,ix[:,:,None].expand(n,width,h.shape[-1]))
    q=a.q_norm(a.q_proj(query).reshape(n,width,c.heads,c.head_dim))
    k=a.k_norm(a.k_proj(x).reshape(n,t,c.kv_heads,c.head_dim));v=a.v_proj(x).reshape(n,t,c.kv_heads,c.head_dim)
    def rotary(z,pos):
        angle=pos.float()[:,:,None]*a.inv;angle=torch.cat([angle,angle],-1)[:,:,None,:]
        return z*angle.cos().to(z.dtype)+torch.cat([-z[...,c.head_dim//2:],z[...,:c.head_dim//2]],-1)*angle.sin().to(z.dtype)
    q=rotary(q,ix);k=rotary(k,positions[None,:].expand(n,-1))
    allowed=positions[None,None,:]<=ix[:,:,None]
    value=F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2).repeat_interleave(c.heads//c.kv_heads,1),v.transpose(1,2).repeat_interleave(c.heads//c.kv_heads,1),attn_mask=allowed[:,None])
    value=value.transpose(1,2).reshape(n,width,-1)
    z=h[active].gather(1,ix[:,:,None].expand_as(value))+a.o_proj(value)
    z=z+block.mlp(block.post_attention_layernorm(z))
    row=active[:,None].expand(n,width)[valid];col=ix[valid]
    return result.index_put((row,col),z[valid])


def physical_work(selected):
    counts=selected.sum(1);n=int((counts>0).sum());width=int(counts.max())
    return dict(logical_queries=int(counts.sum()),padded_queries=n*width,kv_tokens=n*selected.shape[1])
