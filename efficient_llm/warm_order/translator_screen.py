"""Translator screen: does a tied translator T make middle blocks swappable cheaply?

Design: docs/TRANSLATOR_SCREEN_20261123.md. 2x2x2 factorial, all from scratch:
  translator: 'tied' (E->T->Ma->T->Mb->T->H, T shared) or 'slot' (five distinct blocks,
              middles at positions 1 and 3, per-slot translators at 0, 2, 4)
  order:      'fixed' (always Ma before Mb) or 'mix' (alternate Ma/Mb order every step)
  mix:        'residual' (native blocks) or 'attnres' (Kimi AttnRes over sublayers,
              one pseudo-query per stored block sublayer, shared across reuses)
Stored blocks: tied uses blocks 0 (T), 1 (Ma), 2 (Mb); slot uses blocks 0..4.
"""
import torch
from torch import nn
from torch.nn import functional as F

from efficient_llm.dot_pool.model import Block, Config as BlockConfig
from .embedding_anchor import EmbeddingAnchor

ROUTES = {
    'tied': ((0, 1, 0, 2, 0), (0, 2, 0, 1, 0)),
    'slot': ((0, 1, 2, 3, 4), (0, 3, 2, 1, 4)),
}
ARMS = {f'{t}_{o}_{m}': dict(translator=t, order=o, mix=m)
        for t in ('tied', 'slot') for o in ('fixed', 'mix') for m in ('residual', 'attnres')}
STEPS = 12288


def train_route(arm, step):
    native, swapped = ROUTES[ARMS[arm]['translator']]
    return swapped if ARMS[arm]['order'] == 'mix' and step % 2 else native


class TranslatorScreen(EmbeddingAnchor):
    def __init__(self, config, arm):
        super().__init__(config)  # embed, blocks 0..3, head: identical base init for every arm
        if arm not in ARMS:
            raise ValueError('Unknown translator arm')
        self.arm, self.spec = arm, ARMS[arm]
        if self.spec['translator'] == 'slot':
            bc = BlockConfig(d=config.d, heads=config.heads, kv_heads=config.kv_heads,
                             head_dim=config.head_dim, ff=config.ff)
            extra = Block(bc, True)
            for module in extra.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=.02)
            self.blocks.append(extra)
        if self.spec['mix'] == 'attnres':
            self.attnres_query = nn.Parameter(torch.zeros(2 * len(self.blocks) + 1, config.d))

    def _mix(self, sources, index):
        values = torch.stack(sources, dim=0)
        keys = F.rms_norm(values.float(), (self.c.d,), eps=1e-6)
        logits = (keys * self.attnres_query[index].float()).sum(-1)
        return (torch.softmax(logits, dim=0).to(values.dtype).unsqueeze(-1) * values).sum(0)

    def hidden(self, x, path, *, alpha=1., donor=None, skip=None):
        from .embedding_anchor_runtime import anchor_device
        if alpha != 1. or donor is not None or skip is not None:
            raise ValueError('Translator screen takes no anchor, donor or skip options')
        if x.device.type != anchor_device() or self.embed.weight.device.type != anchor_device():
            raise RuntimeError('Explicit matching anchor device scope required')
        route = tuple(path)
        if route not in ROUTES[self.spec['translator']]:
            raise ValueError('Route not registered for this arm')
        pos = torch.arange(x.shape[1], device=x.device).expand_as(x)
        seg = torch.zeros_like(x)
        e = self.embed(x)
        if self.spec['mix'] == 'residual':
            h = e
            for identity in route:
                h = self.blocks[identity](h, pos, seg)
            return h
        sources = [e]
        for identity in route:
            block = self.blocks[identity]
            sources.append(block.self_attn(block.input_layernorm(self._mix(sources, 2 * identity)), pos, seg))
            sources.append(block.mlp(block.post_attention_layernorm(self._mix(sources, 2 * identity + 1))))
        return self._mix(sources, 2 * len(self.blocks))
