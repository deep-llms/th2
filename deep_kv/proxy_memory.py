"""P7: isolated causal estimators, joint native/proxy attention and relational KL.

No model/data RNG is consumed by query sampling. Attention has one softmax;
the FA4 adapter changes entry order, never the mathematical attention problem.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, rotate_half


def eligible_queries(documents):
    eligible = torch.zeros_like(documents, dtype=torch.bool)
    eligible[:, 1:] = documents[:, 1:] == documents[:, :-1]
    return eligible


class MemoryPlan:
    def __init__(self, documents, *, step=0, sequence_indices=None, ems=None, sample_queries=True, convolution=True):
        self.documents, self.ems = documents, ems
        self.conv_masks = tuple((documents[:, lag:] == documents[:, :-lag]).unsqueeze(-1)
                                for lag in range(1, min(4, documents.shape[1]))) if convolution else ()
        self._typed_conv_masks = {}
        self._relational_layout = None
        self.queries = []
        if not sample_queries:
            return  # No GPU-to-CPU sampling synchronization during target bootstrap/inference.
        if sequence_indices is None:
            sequence_indices = range(len(documents))
        if len(sequence_indices) != len(documents):
            raise ValueError('P7 requires one sampling index per sequence')
        # Small metadata copied once per forward, reused by all proxy layers.
        eligible = eligible_queries(documents).cpu()
        selections = []
        for positions, index in zip(eligible, sequence_indices):
            indices = positions.nonzero().flatten()
            generator = torch.Generator(device='cpu')
            generator.manual_seed((int(step)*6364136223846793005 + int(index)*1442695040888963407) % (2**63-1))
            if len(indices) > 256:
                indices = indices[torch.randperm(len(indices), generator=generator)[:256]]
            selections.append(indices)
        # One transfer per microbatch, with views for the individual sequences.
        self.queries = list(torch.cat(selections).to(documents.device).split([len(q) for q in selections]))

    def convolution_masks(self, dtype):
        # One conversion per forward/dtype, shared by layers and checkpoint replay.
        if dtype not in self._typed_conv_masks:
            self._typed_conv_masks[dtype] = tuple(mask.to(dtype=dtype) for mask in self.conv_masks)
        return self._typed_conv_masks[dtype]

    def relational_layout(self):
        """Immutable padded indices/masks shared by every proxy layer and replay."""
        if self._relational_layout is None:
            if len(self.queries) != len(self.documents):
                raise ValueError('Relational loss requires sampled queries')
            counts = self.documents.new_tensor([len(q) for q in self.queries])
            size = max(1, max(len(q) for q in self.queries))
            indices = torch.stack([F.pad(q, (0,size-len(q))) for q in self.queries])
            valid = torch.arange(size, device=indices.device)[None] < counts[:,None]
            positions = torch.arange(self.documents.shape[1], device=indices.device)
            allowed = ((positions[None,None] < indices[:,:,None]) &
                       (self.documents.gather(1,indices)[:,:,None] == self.documents[:,None]))
            # Padding gets one harmless candidate to avoid all-masked softmax.
            # Its KL is explicitly removed below, including its gradient.
            allowed = allowed | ((~valid)[:,:,None] & (positions == 0)[None,None])
            # Cache the actual masked_fill mask and FP32 denominator too;
            # autograd can share this mask instead of saving copies per layer.
            self._relational_layout = indices, valid, ~allowed, counts.float()
        return self._relational_layout


class MemoryHead(nn.Module):
    def __init__(self, config, settings, *, native_values=False):
        super().__init__()
        self.w_in = nn.Linear(config.hidden_size, settings.width, bias=False)
        self.conv = nn.Parameter(torch.empty(4, settings.width))
        self.w_out = nn.Linear(settings.width, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, 2*config.head_dim, bias=False)
        self.v_proj = None if native_values else nn.Linear(config.hidden_size, 2*config.head_dim, bias=False)
        self.k_norm = Qwen3RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=config.initializer_range)
        # Depthwise Conv1d's default fan-in is the kernel length, four.
        nn.init.uniform_(self.conv, -1/math.sqrt(4), 1/math.sqrt(4))

    def estimate(self, u, plan):
        x = self.w_in(u.detach())  # Structural isolation, not a zero upstream VJP.
        c = x*self.conv[0]
        for lag, same in enumerate(plan.convolution_masks(c.dtype), start=1):
            # c owns storage and is not saved by the multiply's backward.
            # Preserve lag/addition order without padded full-length temporaries.
            c[:, lag:].add_(x[:, :-lag]*self.conv[lag]*same)
        if plan.ems is not None:
            c = c + (plan.ems.apply(x, 1)+plan.ems.apply(x, 2)).to(c.dtype)
        return self.w_out(F.silu(c))

    def entries(self, prediction, rotary, native_values):
        with torch.autocast(prediction.device.type, enabled=False):
            # Opt-in supervised adaptation keeps identical forward values while
            # allowing task gradients into the pretrained estimator. Pretraining
            # and likelihood evaluation retain the original routing by default.
            value = (prediction if getattr(self, 'task_finetuning', False) else prediction.detach()).float()
            p = (value*torch.rsqrt(value.square().mean(-1, keepdim=True)+1e-6)).to(prediction.dtype)
        shape = (*p.shape[:-1], 2, -1)
        k = self.k_norm(self.k_proj(p).view(shape)).transpose(1, 2)
        # Only keys need rotating; calling the Q/K helper with (k,k) did twice
        # the same work and discarded one result.
        cos,sin = (x.unsqueeze(1) for x in rotary)
        k = k*cos+rotate_half(k)*sin
        v = native_values if self.v_proj is None else self.v_proj(p).view(shape).transpose(1, 2)
        return k, v


class SimpleMemoryHead(MemoryHead):
    """P4-iso's tokenwise estimator with P7's detached memory projections."""
    def __init__(self, config, settings):
        nn.Module.__init__(self)
        self.w1 = nn.Linear(config.hidden_size, settings.width, bias=False)
        self.w2 = nn.Linear(settings.width, config.hidden_size, bias=False)
        for module in (self.w1, self.w2):
            nn.init.normal_(module.weight, std=config.initializer_range)

    def initialize_projections(self, config):
        # Called after ALL estimators: preserve P4-iso's estimator RNG sequence.
        self.k_proj = nn.Linear(config.hidden_size, 2*config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, 2*config.head_dim, bias=False)
        self.k_norm = Qwen3RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        for module in (self.k_proj, self.v_proj):
            nn.init.normal_(module.weight, std=config.initializer_range)

    def estimate(self, u, plan):
        source = u if getattr(self, 'task_finetuning', False) else u.detach()
        return self.w2(F.silu(self.w1(source)))


def rectangular_mask(mask, disabled=False):
    return torch.cat((mask, torch.zeros_like(mask) if disabled else mask), dim=-1)


def flash_joint_attention(q, k, v, kp, vp, layout, scaling, kernel, observer=None, *, doubled_layout=None):
    """Interleave [native_j, proxy_j], duplicate queries, retain odd outputs.

    Query 2*t+1 sees exactly both entries of tokens j <= t. Even query outputs
    are discarded (zero adjoints). Document lengths/boundaries double. This
    uses the pinned equal-length causal varlen API, without arbitrary masks,
    unequal-length causal alignment assumptions, or separate softmaxes.
    """
    from .fa4 import attention
    query = q.repeat_interleave(2, dim=2)
    key = torch.stack((k, kp), dim=3).flatten(2, 3)
    value = torch.stack((v, vp), dim=3).flatten(2, 3)
    doubled = (layout[0]*2, layout[1]*2) if doubled_layout is None else doubled_layout
    output = attention(query, key, value, doubled, scaling, kernel)
    if observer is not None:
        observer(query, key, value, doubled)
    return output[:, 1::2]


def relational_losses(prediction, target, plan):
    """Return per-sequence KL sums and query counts; empty rows stay differentiable."""
    with torch.autocast(prediction.device.type, enabled=False):
        pred = F.normalize(prediction.float(), dim=-1, eps=1e-6)
        truth = F.normalize(target.detach().float(), dim=-1, eps=1e-6)
        indices, valid, blocked, counts = plan.relational_layout()
        gather = indices[:,:,None].expand(-1,-1,pred.shape[-1])
        target_logits = torch.bmm(truth.gather(1,gather),truth.transpose(1,2))/.1
        pred_logits = torch.bmm(pred.gather(1,gather),pred.transpose(1,2))/.1
        target_logp = target_logits.masked_fill(blocked, -torch.inf).log_softmax(-1)
        pred_logp = pred_logits.masked_fill(blocked, -torch.inf).log_softmax(-1)
        # Avoid 0*(-inf - -inf), including in backward.
        difference = target_logp.masked_fill(blocked, 0)-pred_logp.masked_fill(blocked, 0)
        per_query = (target_logp.exp()*difference).sum(-1)
        return per_query.masked_fill(~valid,0).sum(-1), counts


@torch.no_grad()
def proxy_mass(q, k, kp, documents, scaling, chunk_size=128):
    """Eager FP32 probabilities on bounded query chunks, diagnostics only."""
    repeats = q.shape[1]//k.shape[1]
    keys = torch.cat((k, kp), dim=2).repeat_interleave(repeats, dim=1).float()
    length = q.shape[2]
    positions = torch.arange(length, device=q.device)
    total = q.new_zeros((), dtype=torch.float32)
    with torch.autocast(q.device.type, enabled=False):
        for start in range(0, length, chunk_size):
            end = min(start+chunk_size, length)
            mask = ((positions[None, :] <= positions[start:end, None])[None, None] &
                    (documents[:, None, start:end, None] == documents[:, None, None, :]))
            logits = (q[:, :, start:end].float() @ keys.transpose(-1, -2))*scaling
            probabilities = logits.masked_fill(~rectangular_mask(mask), -torch.inf).softmax(-1)
            total += probabilities[..., length:].sum()
    return torch.stack((total, total.new_tensor(q.shape[0]*q.shape[1]*length)))
