"""Shared-core loop with optional recipient-controlled residual memory."""
import math

import torch
from torch.nn import functional as F

from .embedding_anchor import EmbeddingAnchor


class LoopAnchor(EmbeddingAnchor):
    def __init__(self, config, arm):
        super().__init__(config)
        if arm not in ('serial','memory','static'):
            raise ValueError('Unknown loop arm')
        self.arm=arm
        self.passes=3
        self.memory_enabled=True
        if arm in ('memory','static'):
            self.memory_query=torch.nn.Parameter(torch.zeros(2,config.d))
            self.memory_gain=torch.nn.Parameter(torch.zeros(2,config.d))

    def read_memory(self,h,sources,identity):
        values=torch.stack([F.rms_norm(v.float(),(self.c.d,),eps=1e-6) for v in sources],dim=-2)
        q=self.memory_query[identity-1]
        weights=torch.softmax((values*q).sum(dim=-1)/math.sqrt(self.c.d),dim=-1)
        message=(values*weights.unsqueeze(-1)).sum(dim=-2)
        scale=(h.float().square().mean(dim=-1,keepdim=True)+1e-6).sqrt()
        return h+scale*self.memory_gain[identity-1].tanh()*message

    def hidden(self,x,path,*,alpha=1.,donor=None,skip=None):
        if tuple(path) not in ((0,1,2,3),(0,2,1,3)) or alpha!=1. or donor is not None or skip is not None:
            raise ValueError('Loop experiment requires prelude, two middle blocks and coda')
        if self.passes not in (1,2,3):
            raise ValueError('Pass count outside protocol')
        pos=torch.arange(x.size(1),device=x.device).expand_as(x)
        seg=torch.zeros_like(x)
        e=self.embed(x)
        prelude=self.blocks[0](e,pos,seg)
        h=prelude
        previous=()
        for _ in range(self.passes):
            current=[]
            for identity in path[1:3]:
                if self.arm != 'serial' and self.memory_enabled:
                    # Match the full reader's slot count on later passes.
                    history = (e, prelude) if previous and self.arm == 'static' else previous
                    h=self.read_memory(h,(e,prelude,*history),identity)
                h=self.blocks[identity](h,pos,seg)
                current.append(h)
            previous=tuple(current)
        return self.blocks[3](h,pos,seg)
