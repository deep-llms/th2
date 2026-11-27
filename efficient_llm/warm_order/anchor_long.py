"""Long matched-stack study. Common pretraining uses unchanged historical blocks."""
import math

import torch
from torch.nn import functional as F

from .embedding_anchor import EmbeddingAnchor, Exposure, TRAIN_PATHS

WARM = 16384
DONOR = 4096
STACK = 12288
FORK = WARM + DONOR
END = FORK + STACK
ARMS = ('direct','additive','reader')


def schedule(step):
    if not 768 <= step < END:
        raise ValueError('Step outside continuation')
    if step < WARM:
        phase, path, donor = 'single', (step % 4,), None
    elif step < FORK:
        k = (step - WARM) % 16
        phase, path, donor = 'donor', (k // 4,), k % 4
    else:
        phase, path, donor = 'stack', TRAIN_PATHS[(step-FORK) % len(TRAIN_PATHS)], None
    start = (step-768)*32
    return Exposure(phase,path,1.,donor,start,start+32,.0006)


class LongAnchor(EmbeddingAnchor):
    """Preserve the native hidden coefficient while varying access to the original token."""
    def __init__(self, config, arm):
        super().__init__(config)
        if arm not in ('shared', *ARMS):
            raise ValueError('Unknown architecture')
        self.gate_moments = None
        self.arm = arm
        if arm == 'reader':
            self.query = torch.nn.Parameter(torch.zeros(4,config.d))

    def embedding_gain(self, h, e, identity):
        if self.arm == 'additive':
            return .5
        d = h.shape[-1]
        diff = F.rms_norm(e.float(), (d,), eps=1e-6) - F.rms_norm(h.float(), (d,), eps=1e-6)
        gain = torch.sigmoid((diff * self.query[identity]).sum(-1,keepdim=True)/math.sqrt(d))
        if self.gate_moments is not None:
            v = gain.detach().double()
            self.gate_moments[identity] += torch.stack((v.sum(),v.square().sum(),v.new_tensor(v.numel())))
        return gain

    def hidden(self, x, path, *, alpha=1., donor=None, skip=None):
        if alpha != 1:
            raise ValueError('Long protocol passes the full hidden state')
        if self.arm in ('shared','direct') or len(path)==1:
            return super().hidden(x,path,alpha=alpha,donor=donor,skip=skip)
        if donor is not None or len(set(path))!=len(path) or set(path)-set(range(4)):
            raise ValueError('Invalid stack path')
        e = self.embed(x)
        h = e
        pos = torch.arange(x.shape[1],device=x.device).expand_as(x)
        segments = torch.zeros_like(x)
        for i,identity in enumerate(path):
            if i:
                h = h + self.embedding_gain(h,e,identity)*e
            if identity != skip:
                h = self.blocks[identity](h,pos,segments)
        return h
