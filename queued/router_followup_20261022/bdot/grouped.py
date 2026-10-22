"""Grouped expert projections, preserving the original per-expert parameters.

Only nonempty groups participate in autograd. This is important: a zero weight
gradient is not equivalent to grad=None for AdamW or DDP unused parameters.
The CUDA implementation uses the pinned PyTorch grouped GEMM, not all-expert
dense computation or capacity padding that drops tokens.
"""
import torch
from torch.nn import functional as F


def _grouped_mm(x, weight, offsets):
    if x.device.type != 'cuda' or x.dtype != torch.bfloat16:
        raise ValueError('Grouped expert execution requires CUDA BF16; use reference on CPU/FP32')
    return F.grouped_mm(x, weight.transpose(-2, -1), offs=offsets)


def linear(x, layers, offsets):
    """Rows are already grouped; layers contains only occupied experts in order."""
    if len(layers) == 1:
        return layers[0](x)
    if any(layer.bias is not None for layer in layers):
        raise ValueError('Grouped Qwen projections require bias-free linear layers')
    dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
    # Cast before stacking, avoiding a temporary FP32 copy of all active weights.
    # Unoccupied experts are never stacked, so their parameter gradients stay None.
    weights = torch.stack([layer.weight.to(dtype) for layer in layers])
    with torch.autocast(x.device.type, enabled=False):
        return _grouped_mm(x.to(dtype).contiguous(), weights, offsets)


def queries(step, x, groups):
    sizes = [idx.numel() for _, idx in groups]
    order = torch.cat([idx for _, idx in groups])
    experts = [step.experts[m] for m, _ in groups]
    offsets = torch.tensor(sizes, device=x.device, dtype=torch.int32).cumsum(0, dtype=torch.int32)
    selected = x.flatten(0, 1)[order % (x.shape[0]*x.shape[1])]
    projected = linear(selected, [e.q_proj for e in experts], offsets)
    normalized = torch.cat([step.apply_placement(m,
        e.q_norm(part.view(-1, step.config.num_attention_heads, step.head_dim)).flatten(1), 'query')
        for (m, _), e, part in zip(groups, experts, projected.split(sizes))])
    return order, experts, sizes, offsets, normalized


def outputs(step, h, attention, order, experts, sizes, offsets, groups=None):
    residual = h.flatten(0, 1)[order % (h.shape[0]*h.shape[1])]
    h1 = residual + linear(attention[order], [e.o_proj for e in experts], offsets)
    u = torch.cat([e.mlp_norm(part) for e, part in zip(experts, h1.split(sizes))])
    if step.options.step_film:
        u = u * (1 + step.film_gain.to(u.dtype)) + step.film_bias.to(u.dtype)
    if step.placement_film:
        u = torch.cat([step.apply_placement(m, part, 'mlp')
                       for (m, _), part in zip(groups, u.split(sizes))])
    gate = linear(u, [e.mlp.gate_proj for e in experts], offsets)
    up = linear(u, [e.mlp.up_proj for e in experts], offsets)
    activated = torch.cat([e.mlp.act_fn(part) for e, part in zip(experts, gate.split(sizes))])
    return h1 + linear(activated * up, [e.mlp.down_proj for e in experts], offsets)
