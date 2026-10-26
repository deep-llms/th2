"""Dense execution reference for causal GQA-group selection.

Oracle policies intentionally read actual attention outputs. They are diagnostic
upper references for local reconstruction, not deployable pre-attention routers.
All K/V groups are computed and cached by the unmodified Qwen3 attention module.
"""
from itertools import combinations
import math

import torch


def subset_masks(groups, keep, *, device=None):
    if type(groups) is not int or type(keep) is not int or not 1 <= keep <= groups:
        raise ValueError("Require 1 <= keep <= groups")
    if math.comb(groups, keep) > 4096:
        raise ValueError("Diagnostic subset enumeration is too large")
    choices = list(combinations(range(groups), keep))
    result = torch.zeros(len(choices), groups, dtype=torch.bool, device=device)
    for i, choice in enumerate(choices):
        result[i, list(choice)] = True
    return result


def contributions(attention_output, weight, groups):
    """Return each contiguous GQA group's post-O contribution, excluding bias.

    attention_output: [..., query_heads * head_dim]; weight: [hidden, width].
    Computation is FP32 even when model activations/weights are BF16.
    """
    if weight.ndim != 2 or attention_output.shape[-1] != weight.shape[1]:
        raise ValueError("Attention/output projection shapes differ")
    if groups < 1 or weight.shape[1] % groups:
        raise ValueError("Projection width must divide into whole GQA groups")
    width = weight.shape[1] // groups
    x = attention_output.float().reshape(*attention_output.shape[:-1], groups, width)
    w = weight.float().reshape(weight.shape[0], groups, width)
    return torch.einsum("...gd,hgd->...gh", x, w)


def reconstruction_errors(group_contributions, masks):
    """Exact ||sum of omitted contributions||² using the group Gram matrix."""
    c = group_contributions.float()
    if c.shape[-2] != masks.shape[-1] or masks.ndim != 2:
        raise ValueError("Subset masks and contributions have different group counts")
    omitted = (~masks).float()
    gram = c @ c.transpose(-1, -2)
    return torch.einsum("...ij,si,sj->...s", gram, omitted, omitted).clamp_min(0)


def magnitude_mask(scores, keep):
    if not 1 <= keep <= scores.shape[-1]:
        raise ValueError("Invalid active group count")
    # Stable sort makes exact ties prefer lower group indices.
    selected = scores.argsort(dim=-1, descending=True, stable=True)[..., :keep]
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, selected, True)


def apply_mask(attention_output, mask):
    groups = mask.shape[-1]
    if attention_output.shape[-1] % groups:
        raise ValueError("Mask must select complete contiguous GQA groups")
    shaped = attention_output.reshape(*attention_output.shape[:-1], groups, -1)
    return (shaped * mask.unsqueeze(-1)).reshape_as(attention_output)


class GroupSelection:
    """Temporary hooks at o_proj input; no backbone parameter or KV-cache edits.

    For calibration/dense statistics, collect samples on the dense trajectory.
    For each non-dense policy, choose and apply masks on its own trajectory.
    Statistics are tensor accumulations; no per-token host synchronization.
    """
    POLICIES = {"dense", "static", "raw_norm", "projected_norm", "joint"}

    def __init__(self, model, *, keep, policy, static_masks=None,
                 collect=False, sample_stride=16, chunk_tokens=256):
        if policy not in self.POLICIES or sample_stride < 1 or chunk_tokens < 1:
            raise ValueError("Invalid policy, sample stride, or chunk size")
        if collect and policy != "dense":
            raise ValueError("Reference statistics must use the dense policy")
        config = model.config
        if config.model_type != "qwen3" or config.num_attention_heads % config.num_key_value_heads:
            raise ValueError("Reference currently supports full-attention Qwen3 GQA only")
        if any(getattr(layer.self_attn, "sliding_window", None) is not None for layer in model.model.layers):
            raise ValueError("Sliding attention is outside this diagnostic contract")
        self.groups = config.num_key_value_heads
        self.keep, self.policy = keep, policy
        self.collect, self.stride, self.chunk = collect, sample_stride, chunk_tokens
        self.model = model
        self.masks = subset_masks(self.groups, keep, device=next(model.parameters()).device)
        self.static_masks = static_masks
        if policy == "static":
            expected = (len(model.model.layers), self.groups)
            if static_masks is None or tuple(static_masks.shape) != expected:
                raise ValueError("Static masks must supply every layer")
            if static_masks.dtype != torch.bool or not (static_masks.sum(-1) == keep).all():
                raise ValueError("Static masks have the wrong active-group count")
            self.static_masks = static_masks.to(self.masks.device)
        self.handles = []
        self.stats = {}

    def __enter__(self):
        if self.handles:
            raise RuntimeError("Hooks are already installed")
        for i, layer in enumerate(self.model.model.layers):
            self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(self._hook(i)))
        return self

    def __exit__(self, *unused):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _record(self, index, x, c, errors):
        raw = magnitude_mask(x.float().reshape(-1, self.groups, x.shape[-1] // self.groups).square().sum(-1), self.keep)
        projected = magnitude_mask(c.square().sum(-1), self.keep)
        # Direct errors for norm choices avoid any subset-index search.
        raw_error = (c * (~raw).unsqueeze(-1)).sum(-2).square().sum(-1)
        projected_error = (c * (~projected).unsqueeze(-1)).sum(-2).square().sum(-1)
        energy = c.sum(-2).square().sum(-1)
        row = self.stats.setdefault(index, {
            "candidate_error_sum": torch.zeros(len(self.masks), dtype=torch.float64, device=x.device),
            "totals": torch.zeros(5, dtype=torch.float64, device=x.device),
        })
        row["candidate_error_sum"] += errors.double().sum(0)
        row["totals"] += torch.stack((raw_error.sum(), projected_error.sum(), errors.min(-1).values.sum(), energy.sum(), energy.new_tensor(len(x)))).double()

    def _hook(self, index):
        def hook(module, args):
            x = args[0]
            if len(args) != 1 or x.ndim != 3:
                raise ValueError("Unexpected Qwen3 output-projection input")
            if self.policy == "dense" or self.keep == self.groups:
                if self.collect:
                    samples = x[:, self.stride - 1::self.stride, :].reshape(-1, x.shape[-1])
                    for start in range(0, len(samples), self.chunk):
                        part = samples[start:start+self.chunk]
                        c = contributions(part, module.weight, self.groups)
                        self._record(index, part, c, reconstruction_errors(c, self.masks))
                return None  # Exact identity, including O-projection bias.
            if self.policy == "static":
                return (apply_mask(x, self.static_masks[index]),)
            if self.policy == "raw_norm":
                scores = x.float().reshape(*x.shape[:-1], self.groups, -1).square().sum(-1)
                return (apply_mask(x, magnitude_mask(scores, self.keep)),)
            flattened = x.reshape(-1, x.shape[-1])
            pieces = []
            for start in range(0, len(flattened), self.chunk):
                part = flattened[start:start+self.chunk]
                c = contributions(part, module.weight, self.groups)
                if self.policy == "projected_norm":
                    mask = magnitude_mask(c.square().sum(-1), self.keep)
                else:
                    mask = self.masks[reconstruction_errors(c, self.masks).argmin(-1)]
                pieces.append(apply_mask(part, mask))
            return (torch.cat(pieces).reshape_as(x),)
        return hook
