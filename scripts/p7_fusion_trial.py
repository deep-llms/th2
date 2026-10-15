"""Process-local second-stage P7 candidates; normal training never imports this.

Retain live target gradients and the original separate K/V Parameters. Only
P7-simple-targetgrad is eligible. These are experiments, not production flags.
"""
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class DeferredTarget:
    centered: torch.Tensor
    inverse: torch.Tensor
    clip: float
    target_grad: bool


def normalized_cosine(prediction, centered, inverse, valid, counts, clip, target_grad):
    # Exactly the original normalization followed by cosine_loss, including
    # epsilon/clamp behavior and FP32 accumulation; do not detach a live target.
    from deep_kv.proxy_estimators import cosine_loss
    with torch.autocast(prediction.device.type, enabled=False):
        target = (centered*inverse).clamp(-clip, clip)
        return cosine_loss(prediction, target, valid, target_grad=target_grad, valid_counts=counts)


def packed_entries(self, prediction, rotary, native_values, *, input_grad=False):
    """One K/V GEMM, still using the original separately optimized Parameters."""
    from transformers.models.qwen3.modeling_qwen3 import rotate_half
    with torch.autocast(prediction.device.type, enabled=False):
        value = (prediction if input_grad or getattr(self, 'task_finetuning', False)
                 or getattr(self, 'lm_predictor', False) else prediction.detach()).float()
        p = (value*torch.rsqrt(value.square().mean(-1, keepdim=True)+1e-6)).to(prediction.dtype)
    # No parameter replacement, initialization, cached concatenation or state
    # registration. cat's backward sends gradients to both original weights.
    k, v = F.linear(p, torch.cat((self.k_proj.weight, self.v_proj.weight), dim=0)).chunk(2, dim=-1)
    shape = (*p.shape[:-1], 2, -1)
    k = self.k_norm(k.reshape(shape)).transpose(1, 2)
    cos, sin = (x.unsqueeze(1) for x in rotary)
    k = k*cos+rotate_half(k)*sin
    return k, v.reshape(shape).transpose(1, 2)


@contextmanager
def refinement(mode, backend='inductor'):
    from deep_kv.proxy import ProxyModel
    from deep_kv.proxy_memory import SimpleMemoryHead
    if mode not in ('fused_aux', 'entries', 'kv'):
        raise ValueError(mode)
    original_normalize = ProxyModel.normalize_target
    original_loss = ProxyModel.layer_cosines
    original_entries = SimpleMemoryHead.entries
    # Restore inheritance after the experiment, rather than leave a shadow copy.
    owned_entries = SimpleMemoryHead.__dict__.get('entries')
    try:
        if mode == 'fused_aux':
            fused = torch.compile(normalized_cosine, backend=backend, fullgraph=True, dynamic=False)
            def normalize(self, value, key, *, return_clipped=False, inverse_scale=None,
                          centered=None, target_grad=False):
                if self.arm != 'P7-simple-targetgrad':
                    return original_normalize(self, value, key, return_clipped=return_clipped,
                        inverse_scale=inverse_scale, centered=centered, target_grad=target_grad)
                with torch.set_grad_enabled(torch.is_grad_enabled() and target_grad), torch.autocast(value.device.type, enabled=False):
                    index = self.mean_index[key]
                    if inverse_scale is None:
                        variance = self.sigma2[index]
                        inverse_scale = torch.rsqrt(variance.clamp_min(self.settings.variance_floor*variance.median())+1e-6)
                    if centered is None:
                        centered = (value if target_grad else value.detach()).float()-self.mu[index]
                    deferred = DeferredTarget(centered, inverse_scale, self.settings.target_clip, target_grad)
                    if return_clipped:
                        # Keep diagnostic accounting detached; no persistent state
                        # or mutation is hidden inside a compiled graph.
                        with torch.no_grad():
                            clipped = ((centered*inverse_scale).abs()>self.settings.target_clip).sum()
                        return deferred, clipped
                    return deferred
            def loss(self, layer, u, estimates, target, plan, valid, auxiliary_grad, valid_counts=None):
                if not isinstance(target, DeferredTarget):
                    return original_loss(self, layer, u, estimates, target, plan, valid,
                                         auxiliary_grad, valid_counts=valid_counts)
                counts = valid.sum(-1) if valid_counts is None else valid_counts
                def compute(prediction, centered, inverse):
                    return fused(prediction, centered, inverse, valid, counts, target.clip, target.target_grad)
                with torch.set_grad_enabled(torch.is_grad_enabled() and auxiliary_grad):
                    inputs = (estimates[0], target.centered, target.inverse)
                    if self.training and self.checkpoint_aux and torch.is_grad_enabled():
                        return checkpoint(compute, *inputs, use_reentrant=False)
                    return compute(*inputs)
            ProxyModel.normalize_target, ProxyModel.layer_cosines = normalize, loss
        else:
            # Compile the original normalization/projections/rotation, or its
            # packed GEMM variant. Other head classes remain unchanged.
            fn = packed_entries if mode == 'kv' else original_entries
            SimpleMemoryHead.entries = torch.compile(fn, backend=backend, fullgraph=True, dynamic=False)
        yield
    finally:
        ProxyModel.normalize_target, ProxyModel.layer_cosines = original_normalize, original_loss
        if owned_entries is None:
            if 'entries' in SimpleMemoryHead.__dict__: del SimpleMemoryHead.entries
        else:
            SimpleMemoryHead.entries = owned_entries
