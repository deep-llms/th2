"""Explicit post-block capture, strict-past adapters and differentiable frozen tail.

The Transformers version is pinned because the decoder-layer interface is used
directly. No model or tokenizer is ever retrieved from the network here.
"""
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, apply_rotary_pos_emb

from .protocol import REVISION, TRANSFORMERS_VERSION, validate_config


@dataclass
class Context:
    input_ids: torch.Tensor
    valid: torch.Tensor
    position_ids: torch.Tensor
    segments: torch.Tensor | None = None

    def __post_init__(self):
        shape = self.input_ids.shape
        if len(shape) != 2 or shape[1] < 2:
            raise ValueError("Contexts must have shape [batch, length >= 2]")
        if self.valid.shape != shape or self.position_ids.shape != shape:
            raise ValueError("Context mask and position IDs must match input IDs")
        if self.valid.dtype != torch.bool:
            raise ValueError("valid must be boolean")
        if self.segments is not None and self.segments.shape != shape:
            raise ValueError("segment IDs must match input IDs")

    def allowed(self):
        n = self.input_ids.shape[1]
        causal = torch.ones(n, n, dtype=torch.bool, device=self.input_ids.device).tril()
        mask = causal[None] & self.valid[:, :, None] & self.valid[:, None, :]
        if self.segments is not None:
            mask = mask & (self.segments[:, :, None] == self.segments[:, None, :])
        return mask[:, None]

    def additive_mask(self, dtype):
        # An ignored padding query gets a dummy self edge in the backbone only.
        # No valid query can see it; auxiliary empty-source rows remain zero.
        allowed = self.allowed()
        eye = torch.eye(self.input_ids.shape[1], dtype=torch.bool, device=self.input_ids.device)
        safe = allowed | ((~self.valid)[:, None, :, None] & eye[None, None])
        return torch.zeros_like(safe, dtype=dtype).masked_fill(~safe, torch.finfo(dtype).min)

    def targets(self):
        eligible = self.valid[:, :-1] & self.valid[:, 1:]
        if self.segments is not None:
            eligible = eligible & (self.segments[:, :-1] == self.segments[:, 1:])
        return eligible


class CorrectionAdapter(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.q_norm = Qwen3RMSNorm(hidden_size, eps)
        self.source_norm = Qwen3RMSNorm(hidden_size, eps)
        self.q = nn.Linear(hidden_size, 256, bias=False)
        self.k = nn.Linear(hidden_size, 256, bias=False)
        self.v = nn.Linear(hidden_size, 256, bias=False)
        self.out = nn.Linear(256, hidden_size, bias=False)
        self.gate = nn.Linear(hidden_size, 1, bias=True)
        for projection in (self.q, self.k, self.v):
            nn.init.normal_(projection.weight, std=0.02)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2)

    def forward(self, shallow, source, allowed, rotary, *, diagnostics=False, detach_inputs=True):
        # Pass-A states are never an optimization target, even if supplied with grad.
        query = self.q_norm(shallow.detach() if detach_inputs else shallow)
        source = self.source_norm(source.detach() if detach_inputs else source)
        b, t, _ = query.shape
        def heads(x):
            return x.view(b, t, 2, 128).transpose(1, 2)
        q, k = apply_rotary_pos_emb(heads(self.q(query)), heads(self.k(source)), *rotary)
        v = heads(self.v(source))
        strict = torch.ones(t, t, device=q.device, dtype=torch.bool).tril(-1)
        mask = allowed & strict[None, None]
        nonempty = mask.any(-1, keepdim=True)
        # Avoid all-masked softmax, then explicitly zero those rows.
        # Explicitly disable autocast: casting operands to float alone does
        # not prevent matmul from being cast back to bf16 under autocast.
        with torch.autocast(q.device.type, enabled=False):
            scores = (q.float() @ k.float().transpose(-1, -2)) / (128 ** 0.5)
            scores = scores.masked_fill(~mask, -torch.inf)
            scores = torch.where(nonempty, scores, torch.zeros_like(scores))
            weights = scores.softmax(-1).masked_fill(~mask, 0).to(v.dtype)
        attended = (weights @ v).transpose(1, 2).reshape(b, t, 256)
        gate = self.gate(query).sigmoid()
        correction = (gate * self.out(attended)).to(shallow.dtype)
        if diagnostics:
            return correction, {"attention": weights, "gate": gate, "message": attended}
        return correction


@dataclass
class CleanPass:
    states: dict
    final_hidden: torch.Tensor
    context: Context
    rotary: tuple


class FrozenQwen:
    def __init__(self, model, *, strict_config=True):
        if transformers.__version__ != TRANSFORMERS_VERSION:
            raise ValueError(f"Use transformers=={TRANSFORMERS_VERSION}; got {transformers.__version__}")
        if strict_config:
            validate_config(model.config)
        if model.config.head_dim != 128:
            raise ValueError("Auxiliary RoPE requires the Qwen 128-dimensional heads")
        self.model = model.eval().requires_grad_(False)

    @torch.no_grad()
    def clean(self, context, hooks):
        hooks = tuple(hooks)
        if len(set(hooks)) != len(hooks) or any(not 0 < h < len(self.model.model.layers) for h in hooks):
            raise ValueError("Hooks must be unique post-block indices before the final block")
        states, handles = {}, []
        def capture(h):
            def callback(module, args, output):
                states[h] = output.detach()
            return callback
        try:
            for h in hooks:
                handles.append(self.model.model.layers[h - 1].register_forward_hook(capture(h)))
            output = self.model.model(
                input_ids=context.input_ids, attention_mask=context.additive_mask(self.model.dtype),
                position_ids=context.position_ids, use_cache=False, return_dict=True,
            )
        finally:
            for handle in handles:
                handle.remove()
        rotary = self.model.model.rotary_emb(output.last_hidden_state, context.position_ids)
        return CleanPass(states, output.last_hidden_state.detach(), context, rotary)

    def tail(self, clean, shallow, correction, *, checkpoint_layers=False):
        hidden = clean.states[shallow] + correction
        mask = clean.context.additive_mask(hidden.dtype)
        positions = torch.arange(hidden.shape[1], device=hidden.device)
        for layer in self.model.model.layers[shallow:]:
            def forward(x, block=layer):
                return block(x, attention_mask=mask, position_ids=clean.context.position_ids,
                             cache_position=positions, position_embeddings=clean.rotary, use_cache=False)
            hidden = checkpoint(forward, hidden, use_reentrant=False) if checkpoint_layers and torch.is_grad_enabled() else forward(hidden)
        return self.model.model.norm(hidden)

    def student(self, context, shallow, adapter):
        """One complete forward with a shallow-only branch, without a teacher.

        This full-context scoring path visits every backbone block exactly once.
        Autoregressive decoding/KV-cache integration is a separate interface.
        """
        if not 0 < shallow < len(self.model.model.layers):
            raise ValueError("Invalid shallow hook")
        captured = {}
        def inject(module, args, hidden):
            rotary = self.model.model.rotary_emb(hidden, context.position_ids)
            delta, diag = adapter(hidden, hidden, context.allowed(), rotary, diagnostics=True)
            captured.update(correction=delta, diagnostics=diag)
            return hidden + delta
        handle = self.model.model.layers[shallow - 1].register_forward_hook(inject)
        try:
            result = self.model.model(input_ids=context.input_ids,
                attention_mask=context.additive_mask(self.model.dtype), position_ids=context.position_ids,
                use_cache=False, return_dict=True)
        finally:
            handle.remove()
        return result.last_hidden_state, captured["correction"], captured["diagnostics"]

    def losses(self, hidden, context, *, chunk_size=128):
        """Per-sequence loss sums/counts; fp32 CE with bounded vocabulary logits.

        Checkpoint each head/CE chunk during training to avoid retaining every
        fp32 [tokens,vocabulary] softmax for the frozen language-model head.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        eligible = context.targets()
        labels = context.input_ids[:, 1:].masked_fill(~eligible, -100)
        sums = torch.zeros(hidden.shape[0], device=hidden.device, dtype=torch.float32)
        for start in range(0, hidden.shape[1] - 1, chunk_size):
            x = hidden[:, start:start + chunk_size]
            y = labels[:, start:start + chunk_size]
            x = x[:, :y.shape[1]]
            def ce(states, targets):
                logits = self.model.lm_head(states).float()
                return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                                       reduction="none").view_as(targets).sum(-1)
            sums = sums + (checkpoint(ce, x, y, use_reentrant=False) if hidden.requires_grad else ce(x, y))
        return sums, eligible.sum(-1)


def load_local(path, device):
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise ValueError(f"Use transformers=={TRANSFORMERS_VERSION}; got {transformers.__version__}")
    path = Path(path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Local pretrained snapshot missing: {path}")
    # Accept a normal HF snapshot or a controller's repo-<revision> directory.
    # This is operator-provided provenance, not a cryptographic weight audit.
    if path.name != REVISION and not path.name.endswith("-" + REVISION):
        raise ValueError(f"Local snapshot must identify locked revision {REVISION}; no fallback allowed")
    config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    validate_config(config)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model, loading = AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16,
        config=config, attn_implementation="eager", output_loading_info=True,
    )
    # HF can otherwise initialize missing parameters and merely print a warning.
    # A partly random model cannot count as the pinned pretrained reference.
    problems = {key: loading.get(key) for key in
                ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if loading.get(key)}
    if problems:
        raise ValueError(f"Pretrained checkpoint did not load exactly: {problems}")
    return FrozenQwen(model.to(device)), tokenizer
