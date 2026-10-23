"""Optional causal early exits; stopped tokens still write every layer's K/V.

The original no-STOP path is untouched. Packed training retains dense attention
rows (inactive queries are discarded); projections and both MLP slices are
compacted. Single-token cached SDPA also compacts attention's batch dimension.
"""
from dataclasses import dataclass, replace
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm, rotate_half, ALL_ATTENTION_FUNCTIONS, eager_attention_forward,
)


@dataclass(frozen=True)
class StopConfig:
    train_random_prob: float = .05
    min_steps: int = 2

    def validate(self, options):
        if (type(self.min_steps) is not int or not 1 <= self.min_steps < options.S
                or not math.isfinite(self.train_random_prob) or not 0 <= self.train_random_prob <= 1):
            raise ValueError('Invalid STOP minimum depth/probability')
        if (not options.reuse or options.curriculum != 'none' or options.gate != 'st'
                or options.placement_film or options.kv_mode != 'step'
                or (options.route_scope != 'mlp' and options.routing != 'fixed')):
            raise ValueError('STOP supports hard reused MLP routing or fixed dense control, without curriculum/placement FiLM')


class StopController(nn.Module):
    def __init__(self, config, options):
        super().__init__()
        self.options = StopConfig(**config.bdot_stop)
        self.options.validate(options)
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = nn.Linear(config.hidden_size, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(options.S))
        nn.init.zeros_(self.head.weight)
        # Fitted separately after trunk adaptation; no LM gradient estimator.
        self.requires_grad_(False)
        self.policy = 'none'
        self.threshold = .5
        self.constant_prob = .05
        self.fixed_depth = options.S
        self.capture_states = False
        self.last_states = []
        self.last_depths = None

    def logits(self, h, step):
        with torch.autocast(device_type=h.device.type, enabled=False):
            return self.head(self.norm(h.float())).squeeze(-1) + self.bias[step]

    def start(self, shape, device, seed, training, depths=None):
        policy = 'random' if training and self.options.train_random_prob else self.policy
        if training and self.policy != 'none':
            raise ValueError('Evaluation STOP overrides cannot be used during trunk training')
        if policy not in ('none', 'random', 'constant', 'fixed', 'learned'):
            raise ValueError('Unknown STOP policy')
        if policy == 'learned' and not 0 <= self.threshold <= 1:
            raise ValueError('STOP threshold must be in [0, 1]')
        if policy == 'constant' and not 0 <= self.constant_prob <= 1:
            raise ValueError('STOP probability must be in [0, 1]')
        steps = self.bias.numel()
        if policy == 'fixed' and not self.options.min_steps <= self.fixed_depth <= steps:
            raise ValueError('Fixed depth is outside the permitted STOP range')
        if depths is not None:
            if (depths.shape != shape or depths.dtype != torch.long
                    or bool(((depths < self.options.min_steps) | (depths > steps)).any())):
                raise ValueError('Explicit STOP depths must be int64 [batch, length] in the permitted range')
        self.last_states = []
        # A separate stream ensures STOP draws cannot change exploration draws.
        generator = torch.Generator(device=device).manual_seed((seed + 7919) % (2**63-1))
        return StopState(self, policy, torch.ones(shape, device=device, dtype=torch.bool),
                         torch.zeros(shape, device=device, dtype=torch.long), generator, depths)


@dataclass
class StopState:
    controller: StopController
    policy: str
    active: torch.Tensor
    depths: torch.Tensor
    generator: torch.Generator
    forced_depths: torch.Tensor | None

    def after(self, h, step):
        self.depths = self.depths + self.active.long()
        ctl = self.controller
        if ctl.capture_states:
            ctl.last_states.append(h.detach())
        if step + 1 < ctl.options.min_steps or step + 1 == ctl.bias.numel():
            return h
        if self.forced_depths is not None:
            stop = self.forced_depths <= step + 1
        elif self.policy == 'fixed':
            stop = torch.full_like(self.active, step + 1 >= ctl.fixed_depth)
        elif self.policy in ('random', 'constant'):
            p = ctl.options.train_random_prob if self.policy == 'random' else ctl.constant_prob
            stop = torch.rand(self.active.shape, device=h.device, generator=self.generator) < p
        elif self.policy == 'learned':
            stop = ctl.logits(h, step).sigmoid() > ctl.threshold
        else:
            return h
        # Replace, never mutate: autograd/checkpoint replay may hold the old mask.
        self.active = self.active & ~stop
        return h


def select_active(layer, value, context, indices):
    """Reuse the original router on active rows; inactive rows make no decisions."""
    sub = replace(context, visits=context.visits[indices], state=context.state[indices],
                  z_losses=[], routes=[], counts=[], entropies=[], saturation=[], stop=None)
    if context.forced is not None:
        sub.forced = context.forced.flatten(1)[:, indices].unsqueeze(1)
    choice, weights = layer.select(value.unsqueeze(0), sub)
    context.visits = context.visits.index_copy(0, indices, sub.visits)
    context.state = context.state.index_copy(0, indices, sub.state)
    routes = torch.full_like(context.stop.active, -1, dtype=torch.long).flatten()
    context.routes.append(routes.index_copy(0, indices, sub.routes[0].flatten()).reshape(context.stop.active.shape))
    for name in ('z_losses', 'counts', 'entropies', 'saturation'):
        getattr(context, name).extend(getattr(sub, name))
    return choice, weights


def empty_decisions(layer, context):
    device = context.visits.device
    context.routes.append(torch.full_like(context.stop.active, -1, dtype=torch.long))
    context.counts.append(torch.zeros(layer.options.S, device=device, dtype=torch.long))
    if layer.training:
        context.saturation.append(torch.zeros(3, device=device, dtype=torch.long))


def held_kv(layer, h, position_embeddings, cache):
    b, t, _ = h.shape
    x = layer.input_layernorm(h)
    k = layer.k_norm(layer.k_proj(x).view(b, t, -1, layer.head_dim)).transpose(1, 2)
    v = layer.v_proj(x).view(b, t, -1, layer.head_dim).transpose(1, 2)
    cos, sin = (a.unsqueeze(1) for a in position_embeddings)
    k = k * cos + rotate_half(k) * sin
    if cache is not None:
        if layer.training:
            raise ValueError('STOP training cannot mutate a KV cache')
        k, v = cache.update(k, v, layer.layer_idx)
    return x, k, v


def execute_active(layer, h, indices, attention_mask, position_embeddings, cache, kwargs):
    b, t, d = h.shape
    expert = layer.experts[layer.step]
    x, k, v = held_kv(layer, h, position_embeddings, cache)
    if not indices.numel():
        return None, None  # K/V were still written; no Q/O/MLP or attention.
    q = expert.q_norm(expert.q_proj(x.flatten(0, 1)[indices]).view(-1, layer.config.num_attention_heads, layer.head_dim))
    cos, sin = [a.expand(b, -1, -1).flatten(0, 1)[indices, None] for a in position_embeddings]
    q = q * cos + rotate_half(q) * sin
    interface = ALL_ATTENTION_FUNCTIONS.get_interface(layer.config._attn_implementation, eager_attention_forward)
    if cache is not None and t == 1 and layer.config._attn_implementation == 'sdpa':
        # Every live query attends its entire own prefix. Batch compaction is
        # exact here; arbitrary packed holes cannot use bottom-right causal FA.
        mask = attention_mask
        if mask is not None and mask.shape[0] != 1:
            mask = mask[indices]
        a, _ = interface(layer, q.unsqueeze(2), k[indices], v[indices], mask,
                         dropout=0., scaling=layer.scaling, sliding_window=None, **kwargs)
        a = a.reshape(indices.numel(), -1)
    else:
        dense_q = q.new_zeros(b*t, layer.config.num_attention_heads, layer.head_dim).index_copy(0, indices, q)
        dense_q = dense_q.view(b, t, layer.config.num_attention_heads, layer.head_dim).transpose(1, 2)
        a, _ = interface(layer, dense_q, k, v, attention_mask, dropout=0.,
                         scaling=layer.scaling, sliding_window=None, **kwargs)
        a = a.reshape(b*t, -1)[indices]
    h1 = h.flatten(0, 1)[indices] + expert.o_proj(a)
    u = expert.mlp_norm(h1)
    if layer.options.step_film:
        u = u * (1 + layer.film_gain.to(u.dtype)) + layer.film_bias.to(u.dtype)
    return h1, u


def stopped_step(layer, h, attention_mask, position_embeddings, cache, context, kwargs):
    """Called only when at least one token is inactive; all-active uses old code."""
    indices = context.stop.active.flatten().nonzero().flatten()
    replay = layer.gradient_checkpointing and layer.training
    def attend(value):
        return execute_active(layer, value, indices, attention_mask, position_embeddings, cache, kwargs)
    h1, u = checkpoint(attend, h, use_reentrant=False) if replay and indices.numel() else attend(h)
    if not indices.numel():
        empty_decisions(layer, context)
        return h
    choice, weights = select_active(layer, u if layer.options.route_scope == 'mlp' else h.flatten(0, 1)[indices], context, indices)
    if layer.options.route_scope == 'mlp':
        home_only = context.forced is None and ((context.routing or layer.options.routing) == 'fixed'
                                                or context.reach == 0 or layer.options.S == 1)
        def mlp(value, gate):
            return layer.mlp_update(value.unsqueeze(0), choice, gate, home_only).squeeze(0)
    else:
        def mlp(value, gate):
            return layer.experts[layer.step].mlp(value)
    update = checkpoint(mlp, u, weights, use_reentrant=False) if replay else mlp(u, weights)
    result = h1 + update
    return h.flatten(0, 1).index_copy(0, indices, result.to(h.dtype)).reshape_as(h)


def forward_with_stop(layer, original_forward, hidden_states, attention_mask=None,
                      position_embeddings=None, past_key_values=None, bdot_context=None, **kwargs):
    state = bdot_context.stop if bdot_context is not None else None
    if state is None:
        return original_forward(hidden_states, attention_mask=attention_mask,
            position_embeddings=position_embeddings, past_key_values=past_key_values,
            bdot_context=bdot_context, **kwargs)
    # One small synchronization per step in this correctness-first pilot.
    if bool(state.active.all()):
        result = original_forward(hidden_states, attention_mask=attention_mask,
            position_embeddings=position_embeddings, past_key_values=past_key_values,
            bdot_context=bdot_context, **kwargs)
    else:
        result = stopped_step(layer, hidden_states, attention_mask, position_embeddings,
                              past_key_values, bdot_context, kwargs)
    return state.after(result, layer.step)
