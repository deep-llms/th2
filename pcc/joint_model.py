"""Differentiable full-backbone experiment; frozen PCC paths remain separate."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .model import CorrectionAdapter, FrozenQwen


class JointQwen(nn.Module):
    def __init__(self, model, arm, *, seed=2901, s=4, d=20, checkpoint_layers=True):
        super().__init__()
        if arm not in ("Base", "Shallow", "Deep"):
            raise ValueError("Unknown joint-training arm")
        if not 0 < s < d < len(model.model.layers):
            raise ValueError("Invalid post-block coordinates")
        if model.config.head_dim != 128 or model.config.attention_dropout != 0:
            raise ValueError("Joint experiment requires 128-d heads and zero dropout")
        self.model = model.float().requires_grad_(True)
        self.arm, self.s, self.d = arm, s, d
        self.checkpoint_layers = checkpoint_layers
        self.adapter = None
        if arm != "Base":
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                self.adapter = CorrectionAdapter(model.config.hidden_size, model.config.rms_norm_eps)
            self.adapter.to(next(model.parameters()).device)
        if model.config.tie_word_embeddings and model.lm_head.weight is not model.model.embed_tokens.weight:
            raise ValueError("Embedding/head tying was lost")

    def forward(self, context):
        # No capture hooks: activation-checkpoint recomputation cannot overwrite
        # captured states or inject a branch a second time during backward.
        hidden = self.model.model.embed_tokens(context.input_ids)
        rotary = self.model.model.rotary_emb(hidden, context.position_ids)
        mask = context.additive_mask(hidden.dtype)
        positions = torch.arange(hidden.shape[1], device=hidden.device)

        def block(index, states):
            layer = self.model.model.layers[index]
            def call(x):
                return layer(x, attention_mask=mask, position_ids=context.position_ids,
                             cache_position=positions, position_embeddings=rotary, use_cache=False)
            return checkpoint(call, states, use_reentrant=False) if self.checkpoint_layers and torch.is_grad_enabled() else call(states)

        for index in range(self.s):
            hidden = block(index, hidden)
        if self.adapter is not None:
            source = hidden
            if self.arm == "Deep":
                for index in range(self.s, self.d):
                    source = block(index, source)
            # Both paths must remain differentiable for joint adaptation.
            delta = self.adapter(hidden, source, context.allowed(), rotary, detach_inputs=False)
            hidden = hidden + delta
        for index in range(self.s, len(self.model.model.layers)):
            hidden = block(index, hidden)
        return self.model.model.norm(hidden)

    def losses(self, hidden, context, *, chunk_size=128):
        # This utility does not freeze or detach and supports fp32 master weights.
        return FrozenQwen.losses(self, hidden, context, chunk_size=chunk_size)
