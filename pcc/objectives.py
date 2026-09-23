"""Shared gradient routing and optimizer recipe for the frozen probe."""
import math
import torch
from torch.nn import functional as F


def correction_scale(corrections, eligible):
    values = corrections[:, :-1][eligible].detach().double()
    if not values.numel():
        raise ValueError("No eligible correction targets")
    scale = values.square().mean().sqrt().item()
    if not math.isfinite(scale) or scale <= 1e-8:
        raise ValueError("Privileged reference has no usable correction scale")
    return scale


def alignment_loss(prediction, teacher, eligible, scale):
    if not math.isfinite(scale) or scale <= 1e-8 or not eligible.any():
        raise ValueError("Invalid correction scale or empty target mask")
    return F.smooth_l1_loss(prediction[:, :-1][eligible].float() / scale,
                            teacher[:, :-1][eligible].detach().float() / scale, beta=1.0)


def optimizer(adapter):
    # FP32 master parameters/moments; forward matmuls use bf16 autocast.
    if any(p.dtype != torch.float32 for p in adapter.parameters()):
        raise ValueError("Keep adapter master parameters in fp32 for AdamW moments")
    matrices, other = [], []
    for name, p in adapter.named_parameters():
        # w_g is conceptually a vector although Linear stores it as [1, D].
        (matrices if p.ndim == 2 and name != "gate.weight" else other).append(p)
    return torch.optim.AdamW([{"params": matrices, "weight_decay": 0.01},
                             {"params": other, "weight_decay": 0.0}],
                            lr=3e-4, betas=(0.9, 0.95), eps=1e-8)


def learning_rate(update, total=128, warmup=7):
    """One-based update; final update reaches 3e-5."""
    if not 1 <= update <= total or not 0 < warmup < total:
        raise ValueError("Invalid scheduler update/budget")
    if update <= warmup:
        return 3e-4 * update / warmup
    phase = (update - warmup) / (total - warmup)
    return 3e-5 + (3e-4 - 3e-5) * (1 + math.cos(math.pi * phase)) / 2
