"""Fixed cross-layer KV sharing and physical SwiGLU channel pruning for Qwen3.

Native decoder order, RMSNorm, RoPE, SDPA, residuals and LM head are preserved.
Each forward owns its KV tensor dictionary; producer tensors are NOT detached.
Only native DynamicCache is supported. Decoder checkpointing is refused because
recomputing a layer independently would invalidate cross-layer tensor lifetimes.
"""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from transformers import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention, Qwen3ForCausalLM, Qwen3MLP, apply_rotary_pos_emb, rotate_half,
)
from transformers.integrations.sdpa_attention import sdpa_attention_forward


def sharing_map(layers, enabled):
    return [i - i % 2 if enabled else i for i in range(layers)]


def validate_map(mapping, layers):
    if (len(mapping) != layers or any(type(s) is not int or s < 0 or s > i
                                    for i, s in enumerate(mapping))
            or any(mapping[s] != s for s in mapping)):
        raise ValueError('KV sources must be earlier/current independent producers')


def forward_context(module, args, kwargs):
    if module.gradient_checkpointing and module.training:
        raise ValueError('Cross-layer KV sharing requires decoder checkpointing disabled')
    cache = kwargs.get('past_key_values')
    if cache is not None and type(cache) is not DynamicCache:
        raise ValueError('KV pilot supports only native DynamicCache')
    if cache is not None:
        signature = tuple(module.config.kv_sources)
        previous = getattr(cache, '_kv_mlp_sources', None)
        if previous is None and cache.get_seq_length() != 0:
            raise ValueError('Cannot import a populated cache from another architecture')
        if previous is not None and previous != signature:
            raise ValueError('Cannot change KV sharing on a populated cache')
        cache._kv_mlp_sources = signature
    kwargs['shared_kv'] = {}
    return args, kwargs


def stamp_cache(module, args, output):
    # Native Qwen can allocate its cache inside forward, after our pre-hook.
    cache = getattr(output, 'past_key_values', None)
    if cache is not None:
        cache._kv_mlp_sources = tuple(module.config.kv_sources)


class SharedAttention(Qwen3Attention):
    def forward(self, hidden_states, position_embeddings, attention_mask,
                past_key_values=None, shared_kv=None, **kwargs):
        if shared_kv is None:
            raise ValueError('Missing per-forward KV state')
        shape = (*hidden_states.shape[:-1], -1, self.head_dim)
        q = self.q_norm(self.q_proj(hidden_states).view(shape)).transpose(1, 2)
        if self.source_idx == self.layer_idx:
            k = self.k_norm(self.k_proj(hidden_states).view(shape)).transpose(1, 2)
            v = self.v_proj(hidden_states).view(shape).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
            if past_key_values is not None:
                k, v = past_key_values.update(k, v, self.layer_idx)
            shared_kv[self.layer_idx] = (k, v)
        else:
            cos, sin = (x.unsqueeze(1) for x in position_embeddings)
            q = q * cos + rotate_half(q) * sin
            k, v = shared_kv[self.source_idx]
        y, _ = sdpa_attention_forward(self, q, k, v, attention_mask,
            dropout=self.attention_dropout if self.training else 0., scaling=self.scaling)
        return self.o_proj(y.reshape(*hidden_states.shape[:-1], -1).contiguous()), None


def select_mlp(mlp, indices):
    """Slice both SwiGLU input projections and matching output columns."""
    indices = torch.as_tensor(indices, device=mlp.down_proj.weight.device, dtype=torch.long)
    width = mlp.gate_proj.out_features
    if (indices.ndim != 1 or not 0 < len(indices) <= width
            or indices.unique().numel() != indices.numel()
            or bool((indices < 0).any()) or bool((indices >= width).any())):
        raise ValueError('Invalid MLP channels')
    for name, axis in [('gate_proj', 0), ('up_proj', 0), ('down_proj', 1)]:
        old = getattr(mlp, name)
        rows, cols = old.weight.shape
        new = nn.Linear(len(indices) if axis == 1 else cols,
                        len(indices) if axis == 0 else rows, bias=old.bias is not None,
                        device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            new.weight.copy_(old.weight.index_select(axis, indices))
            if old.bias is not None:
                new.bias.copy_(old.bias.index_select(0, indices) if axis == 0 else old.bias)
        setattr(mlp, name, new)
    mlp.intermediate_size = len(indices)


class CompressedQwen(Qwen3ForCausalLM):
    def __init__(self, config):
        if config.model_type != 'qwen3' or any(t != 'full_attention' for t in config.layer_types):
            raise ValueError('Pilot supports full-attention Qwen3 only')
        if config.rope_parameters['rope_type'] != 'default':
            raise ValueError('Pilot requires default RoPE')
        config.kv_sources = getattr(config, 'kv_sources', list(range(config.num_hidden_layers)))
        config.mlp_widths = getattr(config, 'mlp_widths', [config.intermediate_size] * config.num_hidden_layers)
        validate_map(config.kv_sources, config.num_hidden_layers)
        if (len(config.mlp_widths) != config.num_hidden_layers
                or any(type(w) is not int or not 0 < w <= config.intermediate_size for w in config.mlp_widths)):
            raise ValueError('Invalid per-layer MLP widths')
        super().__init__(config)
        if config._attn_implementation != 'sdpa':
            raise ValueError('KV pilot uses native SDPA only')
        for i, layer in enumerate(self.model.layers):
            layer.self_attn.__class__ = SharedAttention
            layer.self_attn.source_idx = config.kv_sources[i]
            if config.kv_sources[i] != i:
                layer.self_attn.k_proj = layer.self_attn.v_proj = layer.self_attn.k_norm = None
            if config.mlp_widths[i] != config.intermediate_size:
                local_config = deepcopy(config)
                local_config.intermediate_size = config.mlp_widths[i]
                layer.mlp = Qwen3MLP(local_config)
                layer.mlp.apply(self._init_weights)
        self.model.register_forward_pre_hook(forward_context, with_kwargs=True)
        self.model.register_forward_hook(stamp_cache)
        self.accepts_loss_kwargs = False

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
                targets=None, teacher_hidden=None, teacher_weight=None, kd_chunk=128, **kwargs):
        if teacher_hidden is None:
            if targets is not None:
                raise ValueError('Explicit next-token targets require a distillation teacher')
            return super().forward(input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache, **kwargs)
        if labels is not None or targets is None or teacher_weight is None or past_key_values is not None:
            raise ValueError('Invalid distillation inputs')
        h = self.model(input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, use_cache=False).last_hidden_state
        loss = distill_loss(h, self.lm_head.weight, teacher_hidden, teacher_weight,
                            targets, chunk=kd_chunk)
        return CausalLMOutputWithPast(loss=loss)


def distill_loss(hidden, weight, teacher_hidden, teacher_weight, targets, *, chunk=128):
    """Equal-weight CE + forward KL(T || S), T=1; mean over explicit next tokens.

    Fixed complete sequences/batches only in this pilot. Native DDP averages the
    equal-sized rank means; Trainer handles accumulation. No ignored boundaries.
    """
    if (hidden.shape != teacher_hidden.shape or targets.shape != hidden.shape[:2]
            or chunk < 1 or teacher_hidden.requires_grad or teacher_weight.requires_grad):
        raise ValueError('Incompatible distillation tensors or unfrozen teacher')
    loss = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, hidden.shape[1], chunk):
        logits = F.linear(hidden[:, start:start+chunk], weight).float()
        with torch.no_grad():
            teacher_log = F.linear(teacher_hidden[:, start:start+chunk], teacher_weight).float().log_softmax(-1)
        logp = logits.log_softmax(-1)
        ce = F.cross_entropy(logits.flatten(0, 1), targets[:, start:start+chunk].reshape(-1), reduction='sum')
        kl = F.kl_div(logp, teacher_log, log_target=True, reduction='sum')
        loss = loss + .5 * (ce + kl)
    return loss / targets.numel()


def convert(model, *, share=False, channels=None):
    """Mutate a fresh dense CompressedQwen; never apply twice to a pruned model."""
    n = model.config.num_hidden_layers
    if (model.config.kv_sources != list(range(n))
            or model.config.mlp_widths != [model.config.intermediate_size] * n):
        raise ValueError('Conversion requires a fresh dense model')
    sources = sharing_map(n, share)
    if channels is not None and len(channels) != n:
        raise ValueError('Need one channel list per layer')
    for i, layer in enumerate(model.model.layers):
        a = layer.self_attn
        a.source_idx = sources[i]
        if sources[i] != i:
            a.k_proj = a.v_proj = a.k_norm = None
        if channels is not None:
            select_mlp(layer.mlp, channels[i])
    model.config.kv_sources = sources
    model.config.mlp_widths = [l.mlp.gate_proj.out_features for l in model.model.layers]
    return model


def load_compressed(path, device, dtype=torch.float32):
    return CompressedQwen.from_pretrained(path, local_files_only=True,
        dtype=dtype, attn_implementation='sdpa').to(device)


def allocation(costs, *, prune_units):
    """Greedy marginal NLL allocation with nested per-layer removal choices.

    costs[l][k] is measured loss with k equal-width channel groups removed in l
    only. This is a simple recalibration baseline, not a claim of optimality.
    """
    costs = torch.as_tensor(costs, dtype=torch.float64)
    if costs.ndim != 2 or costs.shape[1] < 2 or not torch.isfinite(costs).all():
        raise ValueError('Need finite per-layer pruning loss curves')
    n, options = costs.shape
    if type(prune_units) is not int or not 0 <= prune_units <= n * (options-1):
        raise ValueError('Infeasible pruning budget')
    removed = [0] * n
    for _ in range(prune_units):
        candidates = [(float(costs[i, k+1] - costs[i, k]), i)
                      for i, k in enumerate(removed) if k+1 < options]
        _, layer = min(candidates)
        removed[layer] += 1
    return removed
