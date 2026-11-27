"""Looped baselines without DOT's T layer, at the same 4 stored blocks and the same block applications
(docs/DOT_T_VS_LOOPS_20261125.md).

  cycle : blocks applied cyclically, B0 B1 B2 B3 B0 B1 B2 ... (cross-layer CYCLE sharing / a looped dense stack)
  prc   : prelude B0, a recurrent pair (B1 B2) applied alternately, coda B3 (middle looping, Huginn-style layout
          without input injection)
  dense : B0 B1 B2 B3 once (the base transformer; used with cross-layer shared heads, shared_heads.py)

Depth is counted in DOT units so sweeps line up with the sandwich: depth d means 2 d + 1 block applications, the same
as a DOT sandwich with d middles (T M ... T). test_depth overrides the depth at evaluation or per training step.

Stronger looped baselines from the literature (f14, docs/DOT_T_ABLATION_20261126.md), as options of cycle / prc:
  inject     : Huginn / Parcae-style input injection - before every block application after the first, the state is
               mixed with the input embedding by one shared adapter, h <- W [h; e] (W starts as [I, 0]: a plain loop).
  step_scale : ITT-style thinking-step encoding - a learned vector per application scales that application's residual
               update, h <- h + phi_t * (block(h) - h) (phi starts at 1: a plain loop); later applications reuse the last.
  inject_add : stable additive input injection (Fan et al. looped transformers / Huginn's re-injection of the input,
               gated) - before every application after the first, h <- h + g * e with a learned per-channel gate g
               (starts at 0: a plain loop). f14's concat-and-project injection diverged; this form cannot blow up at init.
"""
import torch
from torch import nn

from .embedding_anchor import EmbeddingAnchor


class BlockSchedule(EmbeddingAnchor):
    def __init__(self, config, arm, spec):
        super().__init__(config)  # identical base init (embed, B0..B3, head) to every other arm of the seed
        self.arm, self.spec = arm, spec
        self.kind, self.native_depth = spec['schedule'], spec.get('depth')
        self.native_apps = spec.get('apps')  # an exact number of block applications (e.g. 8 = dense4 twice)
        if self.native_apps is not None and (self.kind == 'dense' or self.native_depth is not None or self.native_apps < 3):
            raise ValueError('apps (>= 3) is an alternative to depth for the looped schedules')
        if self.kind not in ('cycle', 'prc', 'dense'):
            raise ValueError('unknown schedule')
        self.test_depth = None
        self.aux = None
        d = config.d
        if (spec.get('inject') or spec.get('step_scale') or spec.get('inject_add')) and self.kind == 'dense':
            raise ValueError('inject / inject_add / step_scale are options of the looped schedules')
        self.inject_gate = nn.Parameter(torch.zeros(d)) if spec.get('inject_add') else None
        if spec.get('ff'):  # f16 control: a wider FFN in every stored block (fresh init), to match a compute budget
            from efficient_llm.dot_pool.model import MLP
            for block in self.blocks:
                block.mlp = MLP(d, spec['ff'])
                for module in block.mlp.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.normal_(module.weight, std=.02)
        if spec.get('huginn_block'):  # f23 T ablation: the same Huginn block (sandwich norms + sqrt(d) embedding), no T
            from efficient_llm.dot_pool.model import enable_sandwich
            for block in self.blocks:
                enable_sandwich(block, d)
        self.inject = None
        if spec.get('inject'):
            self.inject = nn.Linear(2 * d, d, bias=False)
            with torch.no_grad():
                self.inject.weight.zero_()
                self.inject.weight[:, :d].copy_(torch.eye(d))
        self.step_scale = None
        if spec.get('step_scale'):
            self.step_scale = nn.Parameter(torch.ones(len(self.order(self.native_depth)), d))

    def order(self, depth):
        if self.kind == 'dense':  # B0 B1 B2 B3 once (no looping; depth is ignored)
            return [0, 1, 2, 3]
        apps = self.native_apps if depth is None else 2 * depth + 1
        if self.kind == 'cycle':
            return [i % 4 for i in range(apps)]
        return [0] + [1 + i % 2 for i in range(apps - 2)] + [3]

    def hidden(self, x, path, *, alpha=1., donor=None, skip=None):
        from .embedding_anchor_runtime import anchor_device
        if alpha != 1. or donor is not None or skip is not None:
            raise ValueError('looped baselines take no anchor, donor or skip options')
        if x.device.type != anchor_device() or self.embed.weight.device.type != anchor_device():
            raise RuntimeError('Explicit matching anchor device scope required')
        if tuple(path) != (0, 1, 2, 3):
            raise ValueError('looped baselines take the native library path only')
        if x.ndim != 2 or not 1 <= x.shape[1] <= self.c.length:
            raise ValueError('Invalid token matrix')
        pos = torch.arange(x.shape[1], device=x.device).expand_as(x)
        segments = None  # one sequence per row (no packed documents in this input)
        e = h = self.embed(x)
        if self.spec.get('huginn_block'):
            e = h = e * (self.c.d ** .5)
        for i, b in enumerate(self.order(self.test_depth or self.native_depth)):
            if self.inject is not None and i:
                h = self.inject(torch.cat([h, e], -1))
            if self.inject_gate is not None and i:
                h = h + self.inject_gate.to(h.dtype) * e
            out = self._run_block(b, h, pos, segments)
            if self.step_scale is not None:
                out = h + self.step_scale[min(i, self.step_scale.shape[0] - 1)].to(h.dtype) * (out - h)
            h = out
        return h

    def _run_block(self, b, h, pos, segments):
        return self.blocks[b](h, pos, segments)
