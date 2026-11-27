"""Domain screen: do domain-specialized middles create input-dependent routing headroom?

Design: docs/DOMAIN_SCREEN_20261123.md. AttnRes sandwich bodies E->T->M->T->H with stored
blocks 0=T, 1=M_a, 2=M_b (block 3 unused; identical base init across arms).
Arms (training routes per batch; every batch holds one domain, alternating DCLM/FWE):
  single  one shared middle M_a for both domains                    (3 block apps)
  oracle  M_a for DCLM batches, M_b for FineWeb-Edu batches         (3 block apps)
  random  M_a or M_b chosen per batch independent of domain         (3 block apps)
  both    E->T->M_a->T->M_b->T->H for every batch                    (5 block apps)
"""
import torch
from torch import nn
from torch.nn import functional as F

from .embedding_anchor import EmbeddingAnchor

ARMS = ('single', 'oracle', 'random', 'both')
ROUTE_A, ROUTE_B, ROUTE_BOTH = (0, 1, 0), (0, 2, 0), (0, 1, 0, 2, 0)
DOMAINS = ('dclm', 'fwe')
STEPS = 12288


def domain_of(step):
    return DOMAINS[step % 2]


def train_route(arm, step):
    if arm == 'single':
        return ROUTE_A
    if arm == 'both':
        return ROUTE_BOTH
    if arm == 'oracle':
        return ROUTE_A if domain_of(step) == 'dclm' else ROUTE_B
    if arm == 'random':  # deterministic, domain-independent: per-step hash parity
        g = torch.Generator().manual_seed(7919 + step)
        return ROUTE_A if int(torch.randint(0, 2, (1,), generator=g)) == 0 else ROUTE_B
    raise ValueError('Unknown domain arm')


def eval_routes(arm):
    return {'single': {'A': ROUTE_A}, 'both': {'both': ROUTE_BOTH}}.get(arm, {'A': ROUTE_A, 'B': ROUTE_B})


class DomainScreen(EmbeddingAnchor):
    def __init__(self, config, arm):
        super().__init__(config)
        if arm not in ARMS:
            raise ValueError('Unknown domain arm')
        self.arm = arm
        self.attnres_query = nn.Parameter(torch.zeros(2 * len(self.blocks) + 1, config.d))

    def _mix(self, sources, index):
        values = torch.stack(sources, dim=0)
        keys = F.rms_norm(values.float(), (self.c.d,), eps=1e-6)
        logits = (keys * self.attnres_query[index].float()).sum(-1)
        return (torch.softmax(logits, dim=0).to(values.dtype).unsqueeze(-1) * values).sum(0)

    def hidden(self, x, path, *, alpha=1., donor=None, skip=None):
        from .embedding_anchor_runtime import anchor_device
        if alpha != 1. or donor is not None or skip is not None:
            raise ValueError('Domain screen takes no anchor, donor or skip options')
        if x.device.type != anchor_device() or self.embed.weight.device.type != anchor_device():
            raise RuntimeError('Explicit matching anchor device scope required')
        route = tuple(path)
        if route not in (ROUTE_A, ROUTE_B, ROUTE_BOTH):
            raise ValueError('Unregistered route')
        pos = torch.arange(x.shape[1], device=x.device).expand_as(x)
        seg = torch.zeros_like(x)
        sources = [self.embed(x)]
        for identity in route:
            block = self.blocks[identity]
            sources.append(block.self_attn(block.input_layernorm(self._mix(sources, 2 * identity)), pos, seg))
            sources.append(block.mlp(block.post_attention_layernorm(self._mix(sources, 2 * identity + 1))))
        return self._mix(sources, 2 * len(self.blocks))

    @torch.no_grad()
    def prefix_feature(self, x, prefix=32):
        """Causal sequence-level router feature: mean token embedding of the first `prefix` tokens."""
        return self.embed(x[:, :prefix]).float().mean(1)
