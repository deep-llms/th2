"""Tied-translator sandwich (user idea, docs/EMBEDDING_ANCHOR_AND_TIED_TRANSLATOR_20261120.md section 3).

Design: docs/SANDWICH_20261123.md. Stored blocks: blocks[0]=T (translator, tied at every
boundary), blocks[1]=M0, blocks[2]=M1, blocks[3]=M2 (middles); blocks[4] only for the
dense5 control. A 'route' lists stored block ids in execution order, e.g. (0,1,0) is
E->T->M0->T->H and (0,2,0,3,0) is E->T->M1->T->M2->T->H. The no-translator control
uses routes without block 0, e.g. (1,) and (2,3). Native residual blocks throughout.
"""
import torch
from torch import nn

from efficient_llm.dot_pool.model import Block, Config as BlockConfig
from .embedding_anchor import EmbeddingAnchor

T, M0, M1, M2, X = 0, 1, 2, 3, 4
ROUTES = {
    'sandwich_warm': (T, M0, T),
    'sandwich_m1': (T, M1, T),
    'sandwich_m2': (T, M2, T),
    'sandwich_stack': (T, M1, T, M2, T),
    'sandwich_swap': (T, M2, T, M1, T),
    'not_warm': (M0,),
    'not_m1': (M1,),
    'not_m2': (M2,),
    'not_stack': (M1, M2),
    'not_swap': (M2, M1),
    'dense5': (0, 1, 2, 3, 4),
    'dense5_swap': (0, 1, 3, 2, 4),
}


class Sandwich(EmbeddingAnchor):
    def __init__(self, config, with_extra=False):
        super().__init__(config)  # embed, blocks[0..3], shared tokenwise head
        if with_extra:  # dense5 control needs a fifth distinct block
            bc = BlockConfig(d=config.d, heads=config.heads, kv_heads=config.kv_heads,
                             head_dim=config.head_dim, ff=config.ff)
            extra = Block(bc, True)
            for module in extra.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=.02)
            self.blocks.append(extra)

    def hidden(self, x, path, *, alpha=1., donor=None, skip=None):
        from .embedding_anchor_runtime import anchor_device
        if alpha != 1. or donor is not None or skip is not None:
            raise ValueError('Sandwich takes no anchor, donor or skip options')
        if x.device.type != anchor_device() or self.embed.weight.device.type != anchor_device():
            raise RuntimeError('Explicit matching anchor device scope required')
        route = tuple(path)
        if not route or any(type(i) is not int or not 0 <= i < len(self.blocks) for i in route):
            raise ValueError('Invalid route')
        pos = torch.arange(x.shape[1], device=x.device).expand_as(x)
        seg = torch.zeros_like(x)
        h = self.embed(x)
        for identity in route:
            h = self.blocks[identity](h, pos, seg)
        return h

    def set_trainable(self, blocks=None, embed_head=True):
        """blocks=None trains everything; otherwise only the listed stored blocks (+embed/head if asked)."""
        for name, p in self.named_parameters():
            if blocks is None:
                p.requires_grad_(True)
            elif name.startswith('blocks.'):
                p.requires_grad_(int(name.split('.')[1]) in blocks)
            else:
                p.requires_grad_(embed_head)
        return [p for p in self.parameters() if p.requires_grad]
