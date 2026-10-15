"""Experimental fused four-block target preparation, with first-order VJP.

No change to cosine loss, running-moment update or its frozen inverse scale.
FP32 contributions stay shared across overlapping windows so upstream gradient
accumulation still happens before the backbone's BF16 cast. CUDA only.
"""
import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit
def _prepare(A, B, C, D, MU, INV, OUT, MASK, SUM, SQUARE, CLIPPED,
             N: tl.constexpr, H: tl.constexpr, CLIP: tl.constexpr,
             STATS: tl.constexpr, CLIPS: tl.constexpr, SAVE: tl.constexpr,
             R: tl.constexpr, K: tl.constexpr):
    rows = tl.program_id(0)*R+tl.arange(0, R)
    cols = tl.program_id(1)*K+tl.arange(0, K)
    offsets = rows[:, None]*H+cols[None, :]
    valid = (rows[:, None]<N)&(cols[None, :]<H)
    # Same left-associated FP32 summation as the original window.
    raw = tl.load(A+offsets, valid, 0)+tl.load(B+offsets, valid, 0)
    raw = raw+tl.load(C+offsets, valid, 0)
    raw = raw+tl.load(D+offsets, valid, 0)
    centered = raw-tl.load(MU+cols, cols<H, 0)[None, :]
    z = centered*tl.load(INV+cols, cols<H, 0)[None, :]
    tl.store(OUT+offsets, tl.minimum(tl.maximum(z, -CLIP), CLIP), valid)
    if SAVE:
        # torch.clamp's derivative includes values exactly on either boundary.
        tl.store(MASK+offsets, (z>=-CLIP)&(z<=CLIP), valid)
    if STATS:
        x = tl.where(valid, centered, 0.)
        partial = tl.program_id(0)*H+cols
        tl.store(SUM+partial, tl.sum(x, axis=0), cols<H)
        tl.store(SQUARE+partial, tl.sum(x*x, axis=0), cols<H)
    if CLIPS:
        count = tl.sum(tl.sum((valid & (tl.abs(z)>CLIP)).to(tl.int32), axis=0), axis=0)
        tl.store(CLIPPED+tl.program_id(0)*tl.cdiv(H,K)+tl.program_id(1), count)


@triton.jit
def _moments(SUM, SQUARE, OUT_SUM, OUT_SQUARE, H: tl.constexpr,
             PARTS: tl.constexpr, R: tl.constexpr, K: tl.constexpr):
    rows = tl.arange(0, R)
    cols = tl.program_id(0)*K+tl.arange(0, K)
    offsets = rows[:, None]*H+cols[None, :]
    valid = (rows[:, None]<PARTS)&(cols[None, :]<H)
    tl.store(OUT_SUM+cols, tl.sum(tl.load(SUM+offsets,valid,0.),axis=0), cols<H)
    tl.store(OUT_SQUARE+cols, tl.sum(tl.load(SQUARE+offsets,valid,0.),axis=0), cols<H)


@triton.jit
def _backward(GRAD, MASK, INV, OUT, SIZE: tl.constexpr, H: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    gradient = tl.load(GRAD+i, i<SIZE, 0)
    scale = tl.load(INV+i%H)
    live = tl.load(MASK+i, i<SIZE, 0)
    # Match clamp backward then multiply backward, including exact zero outside.
    tl.store(OUT+i, tl.where(live, gradient, 0.)*scale, i<SIZE)


class _Target(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, c, d, mu, inverse, clip, stats, clips):
        values=(a,b,c,d)
        if (a.ndim!=3 or not a.is_cuda or a.dtype!=torch.float32 or not a.numel()
                or any(x.shape!=a.shape or x.device!=a.device or x.dtype!=a.dtype or not x.is_contiguous() for x in values)
                or mu.shape!=(a.shape[-1],) or inverse.shape!=mu.shape
                or any(x.device!=a.device or x.dtype!=a.dtype or not x.is_contiguous() or x.requires_grad for x in (mu,inverse))):
            raise ValueError('Fused targets require four contiguous FP32 CUDA contributions and frozen FP32 channel statistics')
        n,h=a.numel()//a.shape[-1],a.shape[-1]
        rows,cols=32,128
        parts=triton.cdiv(n,rows)
        output=torch.empty_like(a)
        save=any(ctx.needs_input_grad[:4])
        mask=torch.empty_like(a,dtype=torch.bool) if save else output
        sums=torch.empty((parts,h),device=a.device,dtype=a.dtype) if stats else output
        squares=torch.empty_like(sums) if stats else output
        counts=torch.empty(parts*triton.cdiv(h,cols),device=a.device,dtype=torch.int32) if clips else output
        _prepare[(parts,triton.cdiv(h,cols))](a,b,c,d,mu,inverse,output,mask,sums,squares,counts,
            n,h,clip,stats,clips,save,rows,cols,num_warps=4,enable_fp_fusion=False)
        total=total_square=count=None
        if stats:
            total=torch.empty_like(mu);total_square=torch.empty_like(mu)
            _moments[(triton.cdiv(h,8),)](sums,squares,total,total_square,h,parts,
                triton.next_power_of_2(parts),8,num_warps=8,enable_fp_fusion=False)
        if clips:count=counts.sum()
        if save:ctx.save_for_backward(mask,inverse)
        ctx.mark_non_differentiable(*(x for x in (total,total_square,count) if x is not None))
        ctx.set_materialize_grads(False)
        return output,total,total_square,count

    @staticmethod
    @once_differentiable
    def backward(ctx, grad, _sum, _square, _count):
        if grad is None:return (None,)*9
        mask,inverse=ctx.saved_tensors
        grad=grad.contiguous()
        result=torch.empty_like(grad)
        _backward[(triton.cdiv(grad.numel(),1024),)](grad,mask,inverse,result,
            grad.numel(),grad.shape[-1],1024,num_warps=4,enable_fp_fusion=False)
        # Reusing this immutable gradient is valid for four addition operands.
        return result,result,result,result,None,None,None,None,None


def target_window(values, mu, inverse, clip, stats, clips):
    if len(values)!=4:raise ValueError('Fused target requires the original four-block window')
    return _Target.apply(*values,mu,inverse,clip,stats,clips)
