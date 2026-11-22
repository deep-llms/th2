"""CPU-qualified embedding-anchor prototype; no runner or CUDA admission.

Design: docs/EMBEDDING_ANCHOR_PILOT_20261122.md.
The shared tokenwise FFN head is not a Transformer coda.
"""
from dataclasses import dataclass
from itertools import permutations
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from efficient_llm.dot_pool.model import Block, Config as BlockConfig, MLP, RMSNorm
from efficient_llm.warm_order.model import Config


def parity(path):
    return sum(a > b for i, a in enumerate(path) for b in path[i + 1:]) % 2


TRAIN_PATHS = tuple(p for p in permutations(range(4)) if parity(p) == 0)
HELD_OUT_PATHS = tuple(p for p in permutations(range(4)) if parity(p) == 1)


@dataclass(frozen=True)
class Exposure:
    phase: str
    path: tuple
    alpha: float
    donor: int | None
    row_start: int
    row_stop: int
    lr: float


def exposure(step, arm):
    """Zero-based full-lineage step; A/C share rows and phase1 checkpoint."""
    if type(step) is not int or not 0 <= step < 1920 or arm not in ('A', 'C'):
        raise ValueError('Expected step0..1919 and arm A/C')
    lr = .0006 * min((step + 1) / 128, 1.)
    if step < 768:
        phase, path, alpha, donor = 'single', (step % 4,), 1., None
    elif step < 1152:
        k = step - 768
        phase, path, donor = 'donor', ((k % 16) // 4,), k % 4
        alpha = .05 + .45 * k / 383 if arm == 'A' else 1.
    else:
        phase, path, donor = 'stack', TRAIN_PATHS[(step - 1152) % 12], None
        alpha = .5 if arm == 'A' else 1.
    return Exposure(phase, path, alpha, donor, step * 32, (step + 1) * 32, lr)


class EmbeddingAnchor(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        if (c.d != c.heads * c.head_dim or c.heads % c.kv_heads
                or c.head_dim % 2 or not 1 <= c.prefix < c.length):
            raise ValueError('Invalid dimensions or suffix boundary')
        self.c = c
        bc = BlockConfig(d=c.d, heads=c.heads, kv_heads=c.kv_heads,
                         head_dim=c.head_dim, ff=c.ff)
        self.embed = nn.Embedding(c.vocab, c.d)
        self.blocks = nn.ModuleList([Block(bc, True) for _ in range(4)])
        self.head_input_norm = RMSNorm(c.d)
        self.head_ffn = MLP(c.d, c.ff)
        self.head_output_norm = RMSNorm(c.d)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=.02)

    def hidden(self, x, path, *, alpha=1., donor=None, skip=None):
        if x.device.type != 'cpu' or self.embed.weight.device.type != 'cpu':
            raise RuntimeError('Prototype is CPU-only; production admission absent')
        path = tuple(path)
        if (not path or len(set(path)) != len(path)
                or any(type(i) is not int or i not in range(4) for i in path)):
            raise ValueError('Expected nonrepeating block identities0..3')
        if not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError('alpha must be finite and in [0,1]')
        if donor is not None and (type(donor) is not int or donor not in range(4) or len(path) != 1):
            raise ValueError('Donor exposure requires one recipient and donor0..3')
        if skip is not None and skip not in path:
            raise ValueError('Skip must name a block in the path')
        if x.ndim != 2 or not 1 <= x.shape[1] <= self.c.length:
            raise ValueError('Invalid token matrix')
        e = self.embed(x)
        pos = torch.arange(x.shape[1], device=x.device).expand_as(x)
        segments = torch.zeros_like(x)
        h = e
        if donor is not None:
            with torch.no_grad():
                d = self.blocks[donor](e.detach(), pos, segments)
            h = (1 - alpha) * e + alpha * d
        for slot, identity in enumerate(path):
            if slot:
                h = (1 - alpha) * e + alpha * h
            if identity != skip:
                h = self.blocks[identity](h, pos, segments)
        return h

    def readout_hidden(self, h):
        return self.head_output_norm(self.head_ffn(self.head_input_norm(h)))

    def token_losses(self, x, path, **kwargs):
        if x.shape[1] <= self.c.prefix:
            raise ValueError('No scored suffix')
        h = self.readout_hidden(self.hidden(x, path, **kwargs))[:, self.c.prefix - 1:-1]
        y = x[:, self.c.prefix:]
        hh, yy = h.reshape(-1, self.c.d), y.reshape(-1)
        losses = []

        def ce(a, weight, target):
            return F.cross_entropy(F.linear(a, weight).float(), target, reduction='none')

        for start in range(0, len(yy), 1024):
            args = hh[start:start + 1024], self.embed.weight, yy[start:start + 1024]
            losses.append(checkpoint(ce, *args, use_reentrant=False)
                          if self.training and torch.is_grad_enabled() else ce(*args))
        return torch.cat(losses).reshape_as(y)

    def nll(self, x, path, **kwargs):
        return self.token_losses(x, path, **kwargs).mean(1)
