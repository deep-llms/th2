"""Same-pass native K/V targets and strict-past auxiliary attention.

Explicit tensor returns keep activation checkpoint recomputation free of capture
hooks or mutable target caches. Decoder computations follow Qwen3's native order.
"""
import torch
import transformers
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import Qwen3ForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm, apply_rotary_pos_emb, eager_attention_forward, repeat_kv,
)
from pcc.model import Context


class AuxiliaryKV(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.head_dim
        self.k = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.v = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.k_norm = Qwen3RMSNorm(config.head_dim, config.rms_norm_eps)
        self.out = nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=False)
        nn.init.normal_(self.k.weight, std=config.initializer_range)
        nn.init.normal_(self.v.weight, std=config.initializer_range)
        nn.init.zeros_(self.out.weight)
        self.groups = config.num_attention_heads // config.num_key_value_heads

    def forward(self, normalized, rotated_query, rotary, allowed):
        b, t, _ = normalized.shape
        k = self.k_norm(self.k(normalized).view(b, t, -1, self.head_dim)).transpose(1, 2)
        v = self.v(normalized).view(b, t, -1, self.head_dim).transpose(1, 2)
        # The native query is already rotated; only use the returned rotated key.
        _, rotated_k = apply_rotary_pos_emb(rotated_query, k, *rotary)
        strict = allowed & torch.ones(t, t, dtype=torch.bool, device=k.device).tril(-1)
        nonempty = strict.any(-1, keepdim=True)
        # A dummy edge avoids all-masked softmax in every SDPA backend. Its result
        # is explicitly zeroed; it is never an auxiliary source or gradient path.
        dummy = torch.eye(t, dtype=torch.bool, device=k.device)[None, None]
        safe = strict | (~nonempty & dummy)
        dtype = v.dtype
        message = F.scaled_dot_product_attention(
            rotated_query.to(dtype), repeat_kv(rotated_k.to(dtype), self.groups),
            repeat_kv(v, self.groups), attn_mask=safe, dropout_p=0.0,
            scale=self.head_dim ** -0.5)
        message = message.masked_fill(~nonempty, 0)
        correction = self.out(message.transpose(1, 2).reshape(b, t, -1))
        return correction, k, v


class DeepKV(nn.Module):
    def __init__(self, backbone, arm, *, seed=2901, consumer=5, deep_target=21,
                 checkpoint_layers=True, lm_chunk=128):
        super().__init__()
        if transformers.__version__ != "5.9.0":
            raise ValueError("Deep-KV requires transformers==5.9.0; use sampling_b200/train_env")
        if arm not in ("A", "B", "C", "D") or not 1 <= consumer < deep_target <= len(backbone.model.layers):
            raise ValueError("Invalid arm or 1-based block coordinates")
        if backbone.config.attention_dropout != 0 or any(
                getattr(layer.self_attn, "sliding_window", None) is not None for layer in backbone.model.layers):
            raise ValueError("Pilot requires full attention and zero dropout")
        self.backbone = backbone.float().requires_grad_(True)
        self.arm, self.consumer, self.deep_target = arm, consumer, deep_target
        self.checkpoint_layers, self.lm_chunk = checkpoint_layers, lm_chunk
        self.aux = None
        if arm != "A":
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed + 1)
                self.aux = AuxiliaryKV(backbone.config)

    @classmethod
    def from_scratch(cls, config, arm, **kwargs):
        # Construct the entire backbone before the branch. Branch initialization
        # never advances the shared backbone's RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(kwargs.get("seed", 2901))
            backbone = Qwen3ForCausalLM(config)
        return cls(backbone, arm, **kwargs)

    def special_block(self, index, hidden, rotary, mask, allowed):
        layer = self.backbone.model.layers[index]
        a = layer.self_attn
        u = layer.input_layernorm(hidden)
        shape = (*u.shape[:-1], -1, a.head_dim)
        q = a.q_norm(a.q_proj(u).view(shape)).transpose(1, 2)
        k = a.k_norm(a.k_proj(u).view(shape)).transpose(1, 2)
        v = a.v_proj(u).view(shape).transpose(1, 2)
        qr, kr = apply_rotary_pos_emb(q, k, *rotary)
        interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.backbone.config._attn_implementation, eager_attention_forward)
        attended, _ = interface(a, qr, kr, v, mask, dropout=0.0,
                               scaling=a.scaling, sliding_window=None)
        attended = a.o_proj(attended.reshape(*u.shape[:-1], -1).contiguous())
        pk = pv = u.new_empty(0)
        if index == self.consumer - 1 and self.aux is not None:
            correction, pk, pv = self.aux(u, qr, rotary, allowed)
            attended = attended + correction
        hidden = hidden + attended
        hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        return hidden, pk, pv, k, v

    def hidden_states(self, context: Context):
        hidden = self.backbone.model.embed_tokens(context.input_ids)
        rotary = self.backbone.model.rotary_emb(hidden, context.position_ids)
        mask, allowed = context.additive_mask(hidden.dtype), context.allowed()
        predicted = target = None
        for i, layer in enumerate(self.backbone.model.layers):
            special = self.aux is not None and (i == self.consumer - 1 or (self.arm == "D" and i == self.deep_target - 1))
            if special:
                def call(x, index=i):
                    return self.special_block(index, x, rotary, mask, allowed)
            else:
                def call(x, block=layer):
                    return block(x, attention_mask=mask, position_ids=context.position_ids,
                                 position_embeddings=rotary, use_cache=False)
            result = checkpoint(call, hidden, use_reentrant=False) if self.checkpoint_layers and self.training else call(hidden)
            if special:
                hidden, pk, pv, k, v = result
                if i == self.consumer - 1:
                    predicted = (pk, pv)
                    if self.arm == "C":
                        target = (k, v)
                if self.arm == "D" and i == self.deep_target - 1:
                    target = (k, v)
            else:
                hidden = result
        return self.backbone.model.norm(hidden), predicted, target

    @staticmethod
    def alignment(predicted, target, valid):
        # Mean over heads/features, sum over nonpadding tokens. The trainer
        # normalizes by global input tokens across ranks and accumulation steps.
        mask = valid[:, None, :, None]
        return tuple(((p.float() - t.detach().float()).abs().masked_fill(~mask, 0)
                      .mean(dim=(1, 3)).sum()) for p, t in zip(predicted, target))

    def forward(self, context: Context):
        hidden, predicted, target = self.hidden_states(context)
        labels = context.input_ids[:, 1:].masked_fill(~context.targets(), -100)
        lm_sum = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, labels.shape[1], self.lm_chunk):
            y = labels[:, start:start + self.lm_chunk]
            x = hidden[:, start:start + y.shape[1]]
            def ce(h, targets):
                logits = self.backbone.lm_head(h).float()
                return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                                       ignore_index=-100, reduction="sum")
            lm_sum = lm_sum + (checkpoint(ce, x, y, use_reentrant=False) if self.training else ce(x, y))
        k_sum = v_sum = lm_sum.new_zeros(())
        if target is not None:
            k_sum, v_sum = self.alignment(predicted, target, context.valid)
        return {"lm_sum": lm_sum, "lm_count": context.targets().sum(),
                "k_sum": k_sum, "v_sum": v_sum, "kv_count": context.valid.sum()}
