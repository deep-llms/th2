"""Optional FA4 varlen adapter for native HF Qwen3 (no model/weight changes).

Boundaries come from source-document IDs, never from EOS or position resets.
The pinned kernel has no dropout; unsupported inputs fail without a fallback.
"""
from functools import lru_cache
from importlib.metadata import version

import torch
from transformers import AttentionInterface

BACKEND = "template_fa4"
FA4_VERSION = "4.0.0b33"


@lru_cache(maxsize=1)
def load_kernel():
    try:
        installed = version("flash-attn-4")
        if installed != FA4_VERSION:
            raise RuntimeError(f"FA4 requires flash-attn-4=={FA4_VERSION}, found {installed}")
        from flash_attn.cute.interface import flash_attn_varlen_func
    except ImportError as error:
        raise RuntimeError(f"Install optional flash-attn-4=={FA4_VERSION} in a compatible CUDA environment; no SDPA fallback") from error
    return flash_attn_varlen_func


def register_backend():
    load_kernel()  # Fail before dataset preprocessing if the dependency is missing.
    AttentionInterface.register(BACKEND, attention_forward)
    return BACKEND


def document_layout(segments):
    """CPU collator only: compute once per batch, avoiding per-layer CUDA syncs."""
    if segments.device.type != "cpu" or segments.ndim != 2 or not segments.numel():
        raise ValueError("FA4 layout requires nonempty CPU [batch, length] document IDs")
    starts = torch.ones_like(segments, dtype=torch.bool)
    starts[:, 1:] = segments[:, 1:] != segments[:, :-1]
    boundaries = starts.flatten().nonzero().flatten()
    cu = torch.cat((boundaries, boundaries.new_tensor([segments.numel()]))).to(torch.int32)
    return cu, int((cu[1:] - cu[:-1]).max())


def validate_kernel_inputs(query, key, value, cu):
    if any(x.device.type != "cuda" for x in (query, key, value, cu)):
        raise ValueError("FA4 requires CUDA Q/K/V and document offsets")
    if value.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("FA4 requires BF16 or FP16 attention (enable mixed precision)")
    if cu.dtype != torch.int32 or cu.ndim != 1:
        raise ValueError("FA4 document offsets must be a one-dimensional int32 tensor")


def attention_forward(module, query, key, value, attention_mask, dropout=0.,
                      scaling=None, sliding_window=None, fa4_cu_seqlens=None,
                      fa4_max_seqlen=None, **kwargs):
    return varlen_attention(load_kernel, module, query, key, value, attention_mask,
        dropout, scaling, sliding_window, fa4_cu_seqlens, fa4_max_seqlen, **kwargs)


def varlen_attention(kernel_loader, module, query, key, value, attention_mask, dropout=0.,
                     scaling=None, sliding_window=None, fa4_cu_seqlens=None,
                     fa4_max_seqlen=None, **kwargs):
    """Shared packed transport; the explicit loader selects the actual kernel.

    The historical fa4_* metadata keys describe document offsets for both FA
    backends. Keeping them avoids changing preprocessing or model semantics.
    """
    if attention_mask is not None or fa4_cu_seqlens is None or fa4_max_seqlen is None:
        raise ValueError("FA4 requires PackedCollator/model_inputs document metadata, without a dense mask")
    if dropout or sliding_window is not None or kwargs.get("use_cache", False):
        raise ValueError("FA4 packed training requires zero dropout, full attention and use_cache=False")
    b, heads, length, dim = query.shape
    if (key.shape != value.shape or key.shape[0] != b or key.shape[2:] != (length, dim)
            or heads % key.shape[1]):
        raise ValueError("FA4 expects equal-length Q/K/V with native grouped-query attention; no KV cache")
    validate_kernel_inputs(query, key, value, fa4_cu_seqlens)
    # QK norm/RoPE may promote Q/K. Match native attention's value precision,
    # retaining autograd through casts and keeping KV heads unexpanded.
    def flatten(x):
        return x.to(value.dtype).transpose(1, 2).reshape(b * length, x.shape[1], dim).contiguous()
    output, _ = kernel_loader()(*map(flatten, (query, key, value)),
        cu_seqlens_q=fa4_cu_seqlens, cu_seqlens_k=fa4_cu_seqlens,
        max_seqlen_q=fa4_max_seqlen, max_seqlen_k=fa4_max_seqlen,
        causal=True, softmax_scale=scaling)
    return output.reshape(b, length, heads, dim), None
