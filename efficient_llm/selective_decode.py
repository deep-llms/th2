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


def _validate(q,k,v,selected,length,window,sinks):
    if q.ndim!=4 or q.shape[2]!=1 or k.ndim!=4 or k.shape!=v.shape or k.stride()!=v.stride():
        raise ValueError('Require single-token query and matching KV layout')
    b,h,_,d=q.shape;g=k.shape[1]
    if (q.dtype not in (torch.float32,torch.float16,torch.bfloat16) or k.dtype!=q.dtype or v.dtype!=q.dtype
        or k.shape[0]!=b or k.shape[3]!=d or h%g or not 1<=length<=k.shape[2]
        or tuple(selected.shape)!=(b,g) or selected.dtype!=torch.bool or window<1 or sinks<0 or d>256
        or any(x.device!=q.device for x in (k,v,selected))):raise ValueError('Unsupported decode shapes/recipe')
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q,k,v)):raise ValueError('Decode is inference-only')
    if not q.is_cuda and os.environ.get('TRITON_INTERPRET')!='1':raise ValueError('CUDA required outside interpreter tests')
    return b,h,d,g


def decode(q,k,v,selected,*,length,window,sinks,scale):
    """Original capacity-sized launch; retained unchanged as the comparison.

    B,H,1,D query; B,G,capacity,D KV; B,G bool global selections.
    length is a host-known cursor, never a device read/synchronization.
    """
    b,h,d,g=_validate(q,k,v,selected,length,window,sinks)
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


@triton.jit
def _selected_group(Selected,b,slot,want_global,G:tl.constexpr,BG:tl.constexpr,
                    M0:tl.constexpr,M1:tl.constexpr):
    groups=tl.arange(0,BG)
    flags=tl.load(Selected+b*M0+groups*M1,groups<G,0)
    matches=(groups<G)&(flags==want_global)
    ordinal=tl.cumsum(matches.to(tl.int32))-1
    group=tl.min(tl.where(matches & (ordinal==slot),groups,G),0)
    return group,tl.sum(flags.to(tl.int32),0)


@triton.jit(do_not_specialize=['length'])
def _compact_partials(Q,K,V,Selected,A,Stats,Out,length,
                      H:tl.constexpr,G:tl.constexpr,D:tl.constexpr,S:tl.constexpr,KEEP:tl.constexpr,
                      Q0:tl.constexpr,Q1:tl.constexpr,Q3:tl.constexpr,
                      K0:tl.constexpr,K1:tl.constexpr,K2:tl.constexpr,K3:tl.constexpr,
                      M0:tl.constexpr,M1:tl.constexpr,SCALE:tl.constexpr,
                      WINDOW:tl.constexpr,SINKS:tl.constexpr,BLOCK:tl.constexpr,BD:tl.constexpr,BG:tl.constexpr):
    job=tl.program_id(0);b=tl.program_id(1)
    R:tl.constexpr=H//G
    global_job=job<KEEP*R*S
    slot=tl.where(global_job,job//(R*S),(job-KEEP*R*S)//R)
    within=tl.where(global_job,(job//S)%R,(job-KEEP*R*S)%R)
    part=tl.where(global_job,job%S,0)
    group,actual_keep=_selected_group(Selected,b,slot,global_job,G,BG,M0,M1)
    dims=tl.arange(0,BD)
    # Invalid budget/mask combinations produce explicit NaNs, never out-of-range
    # KV reads or uninitialized output. ContextPolicy guarantees this invariant.
    if actual_keep!=KEEP:
        if job==0:
            for h in range(H):
                tl.store(Out+(b*H+h)*D+dims,float('nan'),dims<D)
    else:
        h=group*R+within
        sinks=tl.minimum(SINKS,length);left=tl.maximum(sinks,length-WINDOW)
        count=tl.where(global_job,length,sinks+length-left)
        logical=part*BLOCK+tl.arange(0,BLOCK)
        positions=tl.where(global_job,logical,tl.where(logical<sinks,logical,left+logical-sinks))
        valid=logical<count
        query=tl.load(Q+b*Q0+h*Q1+dims*Q3,dims<D,0).to(tl.float32)
        offsets=b*K0+group*K1+positions[:,None]*K2+dims[None,:]*K3
        keys=tl.load(K+offsets,valid[:,None] & (dims[None,:]<D),0).to(tl.float32)
        scores=tl.sum(keys*query[None,:],1)*SCALE
        scores=tl.where(valid,scores,float('-inf'))
        maximum=tl.max(scores,0);probability=tl.exp(scores-maximum)
        denominator=tl.sum(probability,0)
        values=tl.load(V+offsets,valid[:,None] & (dims[None,:]<D),0).to(tl.float32)
        numerator=tl.sum(probability[:,None]*values,0)
        if global_job:
            packed=(b*KEEP*R+slot*R+within)*S+part
            tl.store(Stats+packed*2,maximum)
            tl.store(Stats+packed*2+1,denominator)
            tl.store(A+packed*D+dims,numerator,dims<D)
        else:
            tl.store(Out+(b*H+h)*D+dims,numerator/denominator,dims<D)


@triton.jit
def _compact_merge(A,Stats,Selected,Out,H:tl.constexpr,G:tl.constexpr,D:tl.constexpr,
                   S:tl.constexpr,KEEP:tl.constexpr,M0:tl.constexpr,M1:tl.constexpr,
                   BS:tl.constexpr,BD:tl.constexpr,BG:tl.constexpr):
    job=tl.program_id(0);b=tl.program_id(1)
    R:tl.constexpr=H//G
    group,actual_keep=_selected_group(Selected,b,job//R,True,G,BG,M0,M1)
    if actual_keep==KEEP:
        splits=tl.arange(0,BS);dims=tl.arange(0,BD)
        packed=(b*KEEP*R+job)*S+splits
        maxima=tl.load(Stats+packed*2,splits<S,float('-inf'))
        sums=tl.load(Stats+packed*2+1,splits<S,0)
        weight=tl.exp(maxima-tl.max(maxima,0))
        denominator=tl.sum(weight*sums,0)
        value=tl.load(A+packed[:,None]*D+dims[None,:],(splits[:,None]<S)&(dims[None,:]<D),0)
        output=tl.sum(value*weight[:,None],0)/denominator
        h=group*R+job%R
        tl.store(Out+(b*H+h)*D+dims,output,dims<D)


def decode_compact(q,k,v,selected,*,length,window,sinks,scale,global_groups):
    """Launch split work only for global heads; local heads write output directly.

    Each batch row must have exactly global_groups selected groups. The caller
    knows this fixed budget without reading a GPU tensor; malformed masks produce
    NaN outputs. Window+sinks is limited to one 256-key tile. No KV gather/copy.
    Experimental opt-in: this does not qualify the prior failed BF16 native gate.
    """
    b,h,d,g=_validate(q,k,v,selected,length,window,sinks)
    if type(global_groups) is not int or not 0<=global_groups<=g or window+sinks>256:
        raise ValueError('Compact decode requires a valid fixed budget and <=256 local keys')
    block=256;s=triton.cdiv(length,block);r=h//g
    partial=torch.empty((b*global_groups*r,s,d),device=q.device,dtype=torch.float32)
    stats=torch.empty((b*global_groups*r,s,2),device=q.device,dtype=torch.float32)
    out=torch.empty((b,1,h,d),device=q.device,dtype=q.dtype)
    _compact_partials[(global_groups*r*s+(g-global_groups)*r,b)](
        q,k,v,selected,partial,stats,out,length,h,g,d,s,global_groups,
        q.stride(0),q.stride(1),q.stride(3),k.stride(0),k.stride(1),k.stride(2),k.stride(3),
        selected.stride(0),selected.stride(1),scale,window,sinks,block,
        triton.next_power_of_2(d),triton.next_power_of_2(g),num_warps=8,enable_fp_fusion=False)
    if global_groups:
        _compact_merge[(global_groups*r,b)](partial,stats,selected,out,h,g,d,s,global_groups,
            selected.stride(0),selected.stride(1),triton.next_power_of_2(s),triton.next_power_of_2(d),
            triton.next_power_of_2(g),num_warps=4)
    return out
