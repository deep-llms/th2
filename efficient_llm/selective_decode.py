"""Inference-only split-K local/global GQA decode, with no historical KV gather.

Original implementation using standard stable softmax partial reductions. See
Triton's fused-attention tutorial for the softmax reduction principle. No training
backward, paged cache, padding, variable batch lengths, or CUDA graph API here.
"""
import os
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['length'])
def _partials(Q,K,V,Selected,A,Stats,length,
              H:tl.constexpr,G:tl.constexpr,D:tl.constexpr,S:tl.constexpr,
              Q0:tl.constexpr,Q1:tl.constexpr,Q3:tl.constexpr,
              K0:tl.constexpr,K1:tl.constexpr,K2:tl.constexpr,K3:tl.constexpr,
              M0:tl.constexpr,M1:tl.constexpr,SCALE:tl.constexpr,
              WINDOW:tl.constexpr,SINKS:tl.constexpr,BLOCK:tl.constexpr,BD:tl.constexpr):
    bh=tl.program_id(0);part=tl.program_id(1);b=bh//H;h=bh%H;g=h//(H//G)
    global_head=tl.load(Selected+b*M0+g*M1)
    sinks=tl.minimum(SINKS,length);left=tl.maximum(sinks,length-WINDOW)
    count=tl.where(global_head,length,sinks+length-left)
    dims=tl.arange(0,BD)
    maximum=float('-inf');denominator=0.;numerator=tl.full((BD,),0,tl.float32)
    if part*BLOCK<count:
        logical=part*BLOCK+tl.arange(0,BLOCK)
        positions=tl.where(global_head,logical,tl.where(logical<sinks,logical,left+logical-sinks))
        valid=logical<count
        query=tl.load(Q+b*Q0+h*Q1+dims*Q3,dims<D,0).to(tl.float32)
        offsets=b*K0+g*K1+positions[:,None]*K2+dims[None,:]*K3
        keys=tl.load(K+offsets,valid[:,None] & (dims[None,:]<D),0).to(tl.float32)
        scores=tl.sum(keys*query[None,:],1)*SCALE
        scores=tl.where(valid,scores,float('-inf'))
        maximum=tl.max(scores,0);probability=tl.exp(scores-maximum)
        denominator=tl.sum(probability,0)
        values=tl.load(V+offsets,valid[:,None] & (dims[None,:]<D),0).to(tl.float32)
        numerator=tl.sum(probability[:,None]*values,0)
    tl.store(Stats+(bh*S+part)*2,maximum)
    tl.store(Stats+(bh*S+part)*2+1,denominator)
    tl.store(A+(bh*S+part)*D+dims,numerator,dims<D)


@triton.jit
def _merge(A,Stats,Out,D:tl.constexpr,S:tl.constexpr,BS:tl.constexpr,BD:tl.constexpr):
    bh=tl.program_id(0);splits=tl.arange(0,BS);dims=tl.arange(0,BD)
    maxima=tl.load(Stats+(bh*S+splits)*2,splits<S,float('-inf'))
    sums=tl.load(Stats+(bh*S+splits)*2+1,splits<S,0)
    weight=tl.exp(maxima-tl.max(maxima,0))
    denominator=tl.sum(weight*sums,0)
    value=tl.load(A+(bh*S+splits[:,None])*D+dims[None,:],(splits[:,None]<S)&(dims[None,:]<D),0)
    output=tl.sum(value*weight[:,None],0)/denominator
    tl.store(Out+bh*D+dims,output,dims<D)


def decode(q,k,v,selected,*,length,window,sinks,scale):
    """B,H,1,D query; B,G,capacity,D KV; B,G bool global selections.

    length is a host-known scalar cache cursor, never a device read/synchronization.
    Cache capacity and strides are stable during measured StaticCache decoding.
    """
    if q.ndim!=4 or q.shape[2]!=1 or k.ndim!=4 or k.shape!=v.shape or k.stride()!=v.stride():
        raise ValueError('Require single-token query and matching KV layout')
    b,h,_,d=q.shape;g=k.shape[1]
    if (q.dtype not in (torch.float32,torch.float16,torch.bfloat16) or k.dtype!=q.dtype or v.dtype!=q.dtype
        or k.shape[0]!=b or k.shape[3]!=d or h%g or not 1<=length<=k.shape[2]
        or tuple(selected.shape)!=(b,g) or selected.dtype!=torch.bool or window<1 or sinks<0 or d>256
        or any(x.device!=q.device for x in (k,v,selected))):raise ValueError('Unsupported decode shapes/recipe')
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q,k,v)):raise ValueError('Decode is inference-only')
    if not q.is_cuda and os.environ.get('TRITON_INTERPRET')!='1':raise ValueError('CUDA required outside interpreter tests')
    block=256;s=triton.cdiv(k.shape[2],block)
    partial=torch.empty((b*h,s,d),device=q.device,dtype=torch.float32)
    stats=torch.empty((b*h,s,2),device=q.device,dtype=torch.float32)
    out=torch.empty((b,1,h,d),device=q.device,dtype=q.dtype)
    _partials[(b*h,s)](q,k,v,selected,partial,stats,length,h,g,d,s,
        q.stride(0),q.stride(1),q.stride(3),k.stride(0),k.stride(1),k.stride(2),k.stride(3),
        selected.stride(0),selected.stride(1),scale,window,sinks,block,triton.next_power_of_2(d),
        num_warps=8,enable_fp_fusion=False)
    _merge[(b*h,)](partial,stats,out,d,s,triton.next_power_of_2(s),triton.next_power_of_2(d),num_warps=4)
    return out
