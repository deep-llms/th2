"""Frozen warm-order intervention primitives; no trainer, loader or runner.

Callers must supply an eval-mode, frozen original warm_order.Model. Production
admission, independent document reservation and GPU parity remain external gates.
Norms travel with their attention/MLP component. Token RoPE is never relabelled.
"""
from dataclasses import dataclass
from itertools import permutations
import math

import torch
from torch.nn import functional as F

from .model import Model


@dataclass(frozen=True)
class Plan:
    # Each pair selects (attention + its input norm, MLP + its input norm).
    # None skips that branch. Repetition and an empty path are intentional.
    slots: tuple

    def __post_init__(self):
        if not isinstance(self.slots, tuple) or len(self.slots) > 4:
            raise ValueError('At most four explicit middle slots')
        for pair in self.slots:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError('Expected immutable attention/MLP pairs')
            if any(i is not None and (type(i) is not int or not 0 <= i < 4) for i in pair):
                raise ValueError('Invalid middle component index')

    @classmethod
    def blocks(cls, path):
        return cls(tuple((i, i) for i in path))


CANONICAL = Plan.blocks((0, 1, 2, 3))
PERMUTATIONS = tuple(Plan.blocks(p) for p in permutations(range(4)))


def _validate(model, x):
    if type(model) is not Model:
        raise ValueError('Only original warm-order Model is supported')
    if any(m.training for m in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Caller must freeze model and set eval mode explicitly')
    if x.ndim != 2 or x.dtype != torch.long or not len(x) or not model.c.prefix < x.shape[1] <= model.c.length:
        raise ValueError('Invalid batch, dtype or conditional suffix length')
    if x.device != model.embed.weight.device or x.min() < 0 or x.max() >= model.c.vocab:
        raise ValueError('Token/device mismatch')
    pos = torch.arange(x.shape[1], device=x.device).expand_as(x)
    return pos, torch.zeros_like(x)


def _slot(model, h, pos, seg, pair):
    before = h
    a, f = pair
    if a is not None:
        h = model.middle[a].attention(h, pos, seg)
    after_attention = h
    if f is not None:
        block = model.middle[f]
        h = h + block.mlp(block.post_attention_layernorm(h))
    return h, after_attention-before, h-after_attention


@torch.no_grad()
def execute(model, x, plan=CANONICAL, capture=False):
    """Functional intervention without modifying modules, weights or mode."""
    pos, seg = _validate(model, x)
    if not isinstance(plan, Plan):
        raise ValueError('Explicit Plan required')
    h = model.stem(model.embed(x), pos, seg)
    records = []
    for pair in plan.slots:
        before = h
        h, attention, mlp = _slot(model, h, pos, seg, pair)
        if capture:
            records.append(dict(input=before, attention_update=attention,
                                mlp_update=mlp, output=h, components=pair))
    hidden = model.norm(model.coda(h, pos, seg))
    return hidden, records


@torch.no_grad()
def common_input(model, x, slot, candidate, scale=1.0):
    """Transplant one whole block at a canonical input, keep suffix identical.

    candidate=None or scale=0 is a real skip. This is a context-specific
    substitution test, not an inventory-preserving permutation. Returned inputs
    and updates remain on the caller's device and must not be bulk exported.
    """
    pos, seg = _validate(model, x)
    if type(slot) is not int or not 0 <= slot < 4:
        raise ValueError('Invalid target slot')
    if candidate is not None and (type(candidate) is not int or not 0 <= candidate < 4):
        raise ValueError('Invalid candidate block')
    if not math.isfinite(scale) or scale < 0:
        raise ValueError('Invalid residual scale')
    h = model.stem(model.embed(x), pos, seg)
    for i in range(slot):
        h = model.middle[i](h, pos, seg)
    before = h
    if candidate is not None and scale != 0:
        transformed = model.middle[candidate](h, pos, seg)
        # Avoid subtract/add roundoff in the exact identity case.
        h = transformed if scale == 1 else h + scale*(transformed-h)
    update = h-before
    for i in range(slot+1, 4):
        h = model.middle[i](h, pos, seg)
    return model.norm(model.coda(h, pos, seg)), dict(input=before, update=update)


@torch.no_grad()
def update_statistics(before, after):
    """Per-token branch magnitude, explicitly flag zero residual denominator."""
    if before.shape != after.shape or before.ndim != 3:
        raise ValueError('Expected matched B,T,D states')
    a, b = before.double(), after.double()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Nonfinite hidden state')
    residual_rms = a.square().mean(-1).sqrt()
    update_rms = (b-a).square().mean(-1).sqrt()
    valid = residual_rms > 0
    ratio = update_rms / torch.where(valid, residual_rms, torch.ones_like(residual_rms))
    return dict(residual_rms=residual_rms, update_rms=update_rms,
                update_to_residual=torch.where(valid, ratio, torch.zeros_like(ratio)),
                ratio_valid=valid)


@torch.no_grad()
def paired_scores(model, x, hidden, reference, chunk=128, target_mask=None):
    """Per-sequence SUMS and counts for paired, token-weighted comparisons.

    KL direction is P_reference || P_intervention over the entire vocabulary.
    Only targets after the observed prefix are scored. All-hidden capture is
    intentionally separate; outputs here are compact CPU tensors.
    """
    _validate(model, x)
    shape = (*x.shape, model.c.d)
    if hidden.shape != shape or reference.shape != shape or hidden.device != x.device or reference.device != x.device:
        raise ValueError('Hidden-state shape/device mismatch')
    if type(chunk) is not int or chunk <= 0:
        raise ValueError('Invalid chunk size')
    targets = x[:, model.c.prefix:]
    if target_mask is None:
        target_mask = torch.ones_like(targets, dtype=torch.bool)
    if target_mask.shape != targets.shape or target_mask.dtype != torch.bool or target_mask.device != x.device:
        raise ValueError('Mask must address scored targets only')
    h = hidden[:, model.c.prefix-1:-1].reshape(-1, model.c.d)
    r = reference[:, model.c.prefix-1:-1].reshape(-1, model.c.d)
    y = targets.reshape(-1)
    collected = {k: [] for k in ('nll', 'reference_nll', 'kl_reference_to_intervention', 'agreement')}
    for i in range(0, len(y), chunk):
        z = F.linear(h[i:i+chunk], model.embed.weight).float()
        zr = F.linear(r[i:i+chunk], model.embed.weight).float()
        if not torch.isfinite(z).all() or not torch.isfinite(zr).all():
            raise ValueError('Nonfinite logits')
        lp, lq = z.log_softmax(-1), zr.log_softmax(-1)
        target = y[i:i+chunk, None]
        collected['nll'].append(-lp.gather(1, target).squeeze(1).double())
        collected['reference_nll'].append(-lq.gather(1, target).squeeze(1).double())
        # Double-precision reduction preserves small signed roundoff; no clipping.
        collected['kl_reference_to_intervention'].append((lq.double().exp()*(lq.double()-lp.double())).sum(-1))
        collected['agreement'].append((z.argmax(-1)==zr.argmax(-1)).double())
    result = {key+'_sum': (torch.cat(parts).reshape_as(targets)*target_mask).sum(1).cpu()
              for key, parts in collected.items()}
    result['token_count'] = target_mask.sum(1).cpu()
    return result
