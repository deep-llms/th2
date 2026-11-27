"""L24 (27 Nov 2026): recurrent DOT, a fair recurrent-depth counterpart of Huginn (docs/FULL_LOOPED_METHODS_20261126.md).

    e = B0(sqrt(d) embed(x));  s = e
    r times:  s = B1(T1(s));  s = B2(T2(s))          (a 2-block core shared over recurrences)
    h = T_out(s) -> DOT head FFN -> tied unembedding

Same three attention blocks, head FFN and per-recipient translators as DOT d8, but one block runs once before the loop
and two blocks recur (Huginn's core size). Each translator T (one per recipient B1, B2, head; shared over recurrences) is
a narrow SwiGLU MLP with a residual and the DOT-native RMSNorm on its output:
    T(s)    = norm_out(s + MLP(norm_in(s)))                       direction A (only the training recipe changes)
    T(s, e) = norm_out(s + MLP([norm_in(s), norm_e(e)]))          direction B, inject=True (T also sees the prelude e)
    T(s)    = norm_out(s)                                         control C, t='norm' (no MLP)
The recurrence count is drawn exactly as Huginn draws it (HuginnLM.sample_steps: Poisson-lognormal with mean
`mean_recurrence`, one CPU seed per optimizer step) with the same truncated backprop (gradients through the last
`backprop` recurrences), so the two models see the same depth sequence and the same training compute per recurrence
layout. Not taken from Huginn: random s0, the concat adapter (direction A), sandwich blocks, ELLISAdam.
"""
from contextlib import nullcontext
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from efficient_llm.dot_pool.model import RMSNorm
from .embedding_anchor import EmbeddingAnchor
from .full_methods import HuginnLM


class TMLP(nn.Module):
    """SwiGLU MLP d_in -> f -> d (as dot_pool MLP, with a possibly wider input)."""
    def __init__(self, d_in, d, f):
        super().__init__()
        self.gate_proj = nn.Linear(d_in, f, bias=False)
        self.up_proj = nn.Linear(d_in, f, bias=False)
        self.down_proj = nn.Linear(f, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TMLP2(nn.Module):
    """L32: SwiGLU MLP with two hidden layers d_in -> f -> f -> d (a deeper translator at equal parameters)."""
    def __init__(self, d_in, d, f):
        super().__init__()
        self.gate_proj, self.up_proj = nn.Linear(d_in, f, bias=False), nn.Linear(d_in, f, bias=False)
        self.gate2_proj, self.up2_proj = nn.Linear(f, f, bias=False), nn.Linear(f, f, bias=False)
        self.down_proj = nn.Linear(f, d, bias=False)

    def forward(self, x):
        h = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(F.silu(self.gate2_proj(h)) * self.up2_proj(h))


class Translator(nn.Module):
    """anchor_gate (L33 idea 1): T(s, e) = norm_out(g * n(s) + (1 - g) * n(e) + MLP(...)), g = sigmoid(per-channel
    logit, init 3 -> 0.95): the translator learns how far to pull the state back toward the anchor each time.
    delta (L33 idea 3): the MLP also sees n(delta), delta = the change the previous module made to the state."""
    def __init__(self, d, f, inject, mlp, hidden=1, anchor_gate=False, delta=False):
        super().__init__()
        self.inject, self.anchor_gate, self.use_delta = inject, bool(anchor_gate), bool(delta)
        if self.anchor_gate and not inject:
            raise ValueError('anchor_gate needs the injected anchor e')
        if mlp:
            self.norm_in = RMSNorm(d)
            if inject:
                self.norm_e = RMSNorm(d)
            if self.use_delta:
                self.norm_d = RMSNorm(d)
            self.mlp = (TMLP if hidden == 1 else TMLP2)(d * (1 + bool(inject) + self.use_delta), d, f)
            if self.anchor_gate:
                self.gate_logit = nn.Parameter(torch.full((d,), 3.))
        self.norm_out = RMSNorm(d)

    def forward(self, s, e, code=None, delta=None):
        if 'mlp' not in self._modules:
            return self.norm_out(s)
        ns = self.norm_in(s)
        u = ns
        if code is not None:  # L32 shared T: the recipient's learned code, added to the normalized state (zero init)
            u = u + code.to(u.dtype)
        parts, ne = [u], None
        if self.inject:
            ne = self.norm_e(e)
            parts.append(ne)
        if self.use_delta:
            parts.append(self.norm_d(delta))
        m = self.mlp(torch.cat(parts, -1) if len(parts) > 1 else u)
        if self.anchor_gate:
            g = torch.sigmoid(self.gate_logit).to(ns.dtype)
            return self.norm_out(g * ns + (1 - g) * ne + m)
        return self.norm_out(s + m)


class DotRecurrent(EmbeddingAnchor):
    sample_steps = HuginnLM.sample_steps  # Huginn's sampler verbatim (uses mean_recurrence / backprop only)

    def __init__(self, config, arm, spec):
        super().__init__(config)  # identical base init (embed, B0..B3, head) for every arm of a seed
        self.arm, self.spec, self.f = arm, spec, spec['dotrec']
        f = self.f
        if f['t'] not in ('mlp', 'norm'):
            raise ValueError('Unknown translator kind')
        self.mean_recurrence, self.backprop = f['mean_recurrence'], f['backprop']
        del self.blocks[3]  # B0 runs once (prelude), B1 and B2 recur; B3 is not part of this model
        mlp = f['t'] == 'mlp'
        self.translators = nn.ModuleList(Translator(config.d, f.get('t_ff', 0), bool(f.get('inject')), mlp)
                                         for _ in range(3))  # recipients: B1, B2, head
        for module in self.translators.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=.02)
        if f.get('scaled_init'):  # L31: GPT-2 style residual-output init std / sqrt(2 L), L = applications at r = mean
            layout = f.get('layout')
            apps = 2 * self.mean_recurrence + 2 if layout == 'cycle3' else 1 + 2 * self.mean_recurrence
            with torch.no_grad():
                for block in self.blocks:
                    block.self_attn.o_proj.weight.mul_(1 / math.sqrt(2 * apps))
                    block.mlp.down_proj.weight.mul_(1 / math.sqrt(2 * apps))
        self.passes = 3  # set by the loop-screen trainer; unused (the recurrence count lives in self.f)
        self.aux = None
        self.eval_mode = {}
        self.train_steps = None  # (n_no_grad, k_grad) for the current training step

    def _recur(self, s, e, pos, segments):
        s = self.blocks[1](self.translators[0](s, e), pos, segments)
        return self.blocks[2](self.translators[1](s, e), pos, segments)

    def hidden(self, x, path=None, *, skip=None, segments=None, **kwargs):
        if skip is not None:
            raise ValueError('skip is not defined for the recurrent DOT')
        pos = torch.arange(x.shape[1], device=x.device).expand_as(x)
        e = self.blocks[0](self.embed(x) * math.sqrt(self.c.d), pos, segments)
        s = e
        if self.training and torch.is_grad_enabled():
            n, k = self.train_steps if self.train_steps is not None else self.sample_steps(0)
        else:
            n, k = self.eval_mode.get('steps', self.mean_recurrence), 0
        dt = x.device.type
        # as HuginnLM.hidden: no autocast weight cache in the no-grad recurrences (it would cut later gradients)
        amp = (torch.autocast(dt, dtype=torch.get_autocast_dtype(dt), cache_enabled=False)
               if torch.is_autocast_enabled(dt) else nullcontext())
        with torch.no_grad(), amp:
            for _ in range(n):
                s = self._recur(s, e, pos, segments)
        # full_backprop (A') keeps every recurrence on the gradient path (up to 20 in Huginn's sequence): checkpoint
        # each recurrence so activations fit (same values and gradients, one extra forward per recurrence)
        ckpt = self.f.get('full_backprop') and self.training and torch.is_grad_enabled()
        for _ in range(k):
            s = (checkpoint(self._recur, s, e, pos, segments, use_reentrant=False) if ckpt
                 else self._recur(s, e, pos, segments))
        return self.translators[-1](s, e)  # the head's translator

    def train_loss(self, x, path, step, total):
        n, k = self.sample_steps(step)
        # full_backprop (descriptive arm A'): the same depth sequence, gradients through every recurrence. Without it
        # and without injection, a step with n > 0 detaches the state from the prelude (B0 gets no gradient then).
        self.train_steps = (0, n + k) if self.f.get('full_backprop') else (n, k)
        if self.f.get('fixed_r'):  # L25 diagnostic (as Huginn's f22 'fixed_r'): r = mean_recurrence, full backprop
            self.train_steps = (0, self.mean_recurrence)
        try:
            return self.token_losses(x, path).mean()
        finally:
            self.train_steps = None

    def eval_variants(self):
        r = sorted({1, 2, 3, 4, 6, 8, 16, 32, 2 * self.mean_recurrence} - {self.mean_recurrence})
        return [(f'r{v}', dict(steps=v)) for v in r]


class DotCycle(DotRecurrent):
    """L27: the full DOT trained at Huginn's sampled depth. No prelude: the three middles cycle M1 M2 M3 M1 ... as in
    DOT d8, with a translator before every application (one per recipient, shared over the cycle) that also sees the
    scaled embedding e (the embedding anchor). r recurrences map to Huginn's own application count 2 r + 2 (prelude +
    2 r + coda), so r = 3 is DOT d8's 8 applications; Huginn's sampler and eval_variants are reused unchanged."""

    def __init__(self, config, arm, spec):
        super().__init__(config, arm, spec)  # keeps B0..B2 (B3 removed), the head and the sampler
        f = self.f
        # recipients: M1 = B0, M2 = B1, M3 = B2, head. shared_t (L32, the original DOT T): one translator for every
        # recipient, told the recipient by a learned code (zero init) added to its normalized input
        self.shared_t = bool(f.get('shared_t'))
        self.translators = nn.ModuleList(Translator(config.d, f['t_ff'], bool(f.get('inject')), f['t'] == 'mlp',
                                                    f.get('t_hidden', 1)) for _ in range(1 if self.shared_t else 4))
        if self.shared_t:
            self.recipient_codes = nn.Parameter(torch.zeros(4, config.d))
        for module in self.translators.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=.02)
        if f.get('sandwich_blocks'):  # L29 diagnostic only: Huginn's 4-norm sandwich in the three middles
            from efficient_llm.dot_pool.model import enable_sandwich
            for block in self.blocks:
                enable_sandwich(block, config.d)

    def _t(self, j, s, e):
        if self.shared_t:
            return self.translators[0](s, e, self.recipient_codes[j])
        return self.translators[j](s, e)

    def _app(self, i, s, e, pos, segments):
        j = i % 3
        return self.blocks[j](self._t(j, s, e), pos, segments)

    def hidden(self, x, path=None, *, skip=None, segments=None, **kwargs):
        if skip is not None:
            raise ValueError('skip is not defined for the cycling DOT')
        pos = torch.arange(x.shape[1], device=x.device).expand_as(x)
        e = self.embed(x) * math.sqrt(self.c.d)
        s = e
        if self.training and torch.is_grad_enabled():
            n, k = self.train_steps if self.train_steps is not None else self.sample_steps(0)
        else:
            n, k = self.eval_mode.get('steps', self.mean_recurrence), 0
        total, frozen = 2 * (n + k) + 2, 2 * n  # applications; the first 2 n run without gradient (truncation)
        if not (self.training and torch.is_grad_enabled()):
            frozen = total
        dt = x.device.type
        amp = (torch.autocast(dt, dtype=torch.get_autocast_dtype(dt), cache_enabled=False)
               if torch.is_autocast_enabled(dt) else nullcontext())
        with torch.no_grad(), amp:
            for i in range(frozen):
                s = self._app(i, s, e, pos, segments)
        ckpt = self.f.get('full_backprop') and self.training and torch.is_grad_enabled()
        for i in range(frozen, total):
            s = checkpoint(self._app, i, s, e, pos, segments, use_reentrant=False) if ckpt else self._app(i, s, e, pos, segments)
        return self._t(3, s, e)


class DotPair(DotRecurrent):
    """L30 (user idea): one translator per 2-layer block instead of one per layer, i.e. Huginn's layout with T in place
    of the linear adapter: e = B0(sqrt(d) embed); r times s = B2(B1(T(s, e))); h = T_out(s, e) -> DOT head FFN.
    The two core layers run as an ordinary pre-norm pair; T re-standardizes the state once per recurrence."""

    def __init__(self, config, arm, spec):
        super().__init__(config, arm, spec)  # B0 prelude, B1-B2 core, Huginn's sampler and eval variants
        f = self.f
        self.translators = nn.ModuleList(Translator(config.d, f['t_ff'], bool(f.get('inject')), f['t'] == 'mlp',
                                                    anchor_gate=f.get('anchor_gate'), delta=f.get('delta'))
                                         for _ in range(2))  # recipients: the core pair, the head
        for module in self.translators.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=.02)
        self.use_delta = bool(f.get('delta'))

    def _recur(self, s, e, pos, segments):
        return self.blocks[2](self.blocks[1](self.translators[0](s, e), pos, segments), pos, segments)

    def _recur_delta(self, s, e, d, pos, segments):
        return self.blocks[2](self.blocks[1](self.translators[0](s, e, delta=d), pos, segments), pos, segments)

    def hidden(self, x, path=None, *, skip=None, segments=None, **kwargs):
        if not self.use_delta:
            return super().hidden(x, path, skip=skip, segments=segments, **kwargs)
        if skip is not None:
            raise ValueError('skip is not defined for the recurrent DOT')
        pos = torch.arange(x.shape[1], device=x.device).expand_as(x)
        e = self.blocks[0](self.embed(x) * math.sqrt(self.c.d), pos, segments)
        s, d = e, torch.zeros_like(e)  # no previous module yet: delta 0
        if self.training and torch.is_grad_enabled():
            n, k = self.train_steps if self.train_steps is not None else self.sample_steps(0)
        else:
            n, k = self.eval_mode.get('steps', self.mean_recurrence), 0
        dt = x.device.type
        amp = (torch.autocast(dt, dtype=torch.get_autocast_dtype(dt), cache_enabled=False)
               if torch.is_autocast_enabled(dt) else nullcontext())
        with torch.no_grad(), amp:
            for _ in range(n):
                t = self._recur_delta(s, e, d, pos, segments)
                s, d = t, t - s
        ckpt = self.f.get('full_backprop') and self.training and torch.is_grad_enabled()
        for _ in range(k):
            t = (checkpoint(self._recur_delta, s, e, d, pos, segments, use_reentrant=False) if ckpt
                 else self._recur_delta(s, e, d, pos, segments))
            s, d = t, t - s
        return self.translators[-1](s, e, delta=d)
