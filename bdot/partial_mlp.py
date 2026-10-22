"""Fixed-depth attention and norms, with a fixed core plus one routed MLP slice.

Backbone weights are split, never redrawn. The router sees the current layer's
normalized MLP input; its unit-forward ST gate multiplies only the routed output.
"""
from copy import deepcopy

import torch
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3MLP, rotate_half, ALL_ATTENTION_FUNCTIONS, eager_attention_forward,
)

from .model import RoutedStep


def split_mlp(mlp, config, fraction):
    width = config.intermediate_size
    routed = int(width * fraction)
    if routed != width * fraction or routed < 1:
        raise ValueError('MLP width must divide exactly into the configured channel split')
    if routed == width:
        return None, mlp
    fixed = width - routed
    parts = []
    # Constructing Linear layers must not consume RNG before router initialization.
    with torch.random.fork_rng(devices=[]):
        for start, stop in ((0, fixed), (fixed, width)):
            c = deepcopy(config)
            c.intermediate_size = stop - start
            part = Qwen3MLP(c).to(device=mlp.gate_proj.weight.device,
                                 dtype=mlp.gate_proj.weight.dtype)
            with torch.no_grad():
                part.gate_proj.weight.copy_(mlp.gate_proj.weight[start:stop])
                part.up_proj.weight.copy_(mlp.up_proj.weight[start:stop])
                part.down_proj.weight.copy_(mlp.down_proj.weight[:, start:stop])
            parts.append(part)
    return tuple(parts)


class RoutedMLPStep(RoutedStep):
    def attention_and_input(self, h, attention_mask, position_embeddings, past_key_values, kwargs):
        b, t, _ = h.shape
        expert = self.experts[self.step]
        x = self.input_layernorm(h)
        q = expert.q_norm(expert.q_proj(x).view(b, t, -1, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(b, t, -1, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(b, t, -1, self.head_dim).transpose(1, 2)
        cos, sin = (v.unsqueeze(1) for v in position_embeddings)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        if past_key_values is not None:
            if self.training:
                raise ValueError('Training cannot mutate a KV cache')
            k, v = past_key_values.update(k, v, self.layer_idx)
        interface = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, eager_attention_forward)
        attention, _ = interface(self, q, k, v, attention_mask, dropout=0.,
                                 scaling=self.scaling, sliding_window=None, **kwargs)
        h1 = h + expert.o_proj(attention.reshape(b, t, -1))
        u = expert.mlp_norm(h1)
        if self.options.step_film:
            u = u * (1 + self.film_gain.to(u.dtype)) + self.film_bias.to(u.dtype)
        return h1, u

    def mlp_update(self, u, choice, weights, home_only):
        shape = u.shape
        x = u.flatten(0, 1)
        if home_only:
            routed = self.experts[self.step].mlp(x)
        else:
            ids = choice.flatten()
            order = ids.argsort(stable=True)
            sizes = torch.zeros(self.options.S, dtype=torch.long, device=x.device).scatter_add_(
                0, ids, torch.ones_like(ids)).tolist()
            experts = [self.experts[m].mlp for m, size in enumerate(sizes) if size]
            sizes = [size for size in sizes if size]
            selected = x[order]
            if self.options.expert_backend == 'grouped':
                from .grouped import linear
                offsets = torch.tensor(sizes, device=x.device, dtype=torch.int32).cumsum(0, dtype=torch.int32)
                gate = linear(selected, [e.gate_proj for e in experts], offsets)
                up = linear(selected, [e.up_proj for e in experts], offsets)
                projected = linear(experts[0].act_fn(gate) * up, [e.down_proj for e in experts], offsets)
            else:
                projected = torch.cat([expert(part) for expert, part in zip(experts, selected.split(sizes))])
            routed = torch.empty_like(projected).index_copy(0, order, projected)
            routed = routed * weights.to(routed.dtype)
        update = routed.reshape(shape)
        if self.core_mlp is not None:
            update = self.core_mlp(u) + update
        return update

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None,
                past_key_values=None, bdot_context=None, **kwargs):
        if bdot_context is None or bdot_context.soft:
            raise ValueError('MLP routing requires a hard RouteContext')
        replay = self.gradient_checkpointing and self.training
        if replay and past_key_values is not None:
            raise ValueError('Checkpoint replay cannot mutate a KV cache')
        def attend(h):
            return self.attention_and_input(h, attention_mask, position_embeddings, past_key_values, kwargs)
        h1, u = checkpoint(attend, hidden_states, use_reentrant=False) if replay else attend(hidden_states)
        choice, weights = self.select(u, bdot_context)
        home_only = bdot_context.forced is None and (
            (bdot_context.routing or self.options.routing) == 'fixed'
            or bdot_context.reach == 0 or self.options.S == 1)
        def feedforward(value, gate):
            return self.mlp_update(value, choice, gate, home_only)
        update = checkpoint(feedforward, u, weights, use_reentrant=False) if replay else feedforward(u, weights)
        return h1 + update
