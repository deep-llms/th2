"""Causal suffix LM: order chosen from an observed prefix, never future targets."""
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from efficient_llm.dot_pool.model import Config as BlockConfig, Block, RMSNorm

@dataclass
class Config:
    d: int = 768
    heads: int = 12
    kv_heads: int = 4
    head_dim: int = 64
    ff: int = 2048
    vocab: int = 50304
    prefix: int = 32
    length: int = 512

# Dihedral family: every middle block occurs twice at each position.
ORDERS = tuple(tuple((start + direction*j) % 4 for j in range(4))
               for direction in (1, -1) for start in range(4))

def path_for(arm, step, total):
    """Global minibatch paths eliminate fragmented training kernels; no RNG needed."""
    if arm == 'D': return ORDERS[0]
    if arm not in ('R', 'W'): raise ValueError('Unknown arm')
    depth = 1 if arm == 'W' and step < total//10 else 2 if arm == 'W' and step < total//5 else 4
    return ORDERS[step % 8][:depth]

class Model(nn.Module):
    def __init__(self, c):
        super().__init__(); self.c = c
        bc = BlockConfig(d=c.d, heads=c.heads, kv_heads=c.kv_heads, head_dim=c.head_dim, ff=c.ff)
        self.embed = nn.Embedding(c.vocab, c.d)
        self.stem = Block(bc, True); self.middle = nn.ModuleList([Block(bc, True) for _ in range(4)])
        self.coda = Block(bc, True); self.norm = RMSNorm(c.d)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)): nn.init.normal_(m.weight, std=.02)
    @torch.no_grad()
    def features(self, x):
        # Last shared-stem prefix state; never observe any scored suffix token.
        z = x[:, :self.c.prefix]
        pos = torch.arange(z.shape[1], device=z.device).expand_as(z)
        return self.stem(self.embed(z), pos, torch.zeros_like(z))[:, -1].float().detach()
    def hidden(self, x, path):
        if not path or len(set(path)) != len(path) or any(i not in range(4) for i in path):
            raise ValueError('Expected a nonrepeating middle-layer path')
        if x.shape[1] > self.c.length: raise ValueError('Sequence too long')
        pos = torch.arange(x.shape[1], device=x.device).expand_as(x)
        seg = torch.zeros_like(x); h = self.stem(self.embed(x), pos, seg)
        for i in path: h = self.middle[i](h, pos, seg)
        return self.norm(self.coda(h, pos, seg))
    def nll(self, x, path):
        # Logit at prefix-1 predicts token prefix. Prefix is given, never scored.
        h = self.hidden(x, path)[:, self.c.prefix-1:-1]
        y = x[:, self.c.prefix:]; width = y.shape[1]
        if width <= 0: raise ValueError('No suffix targets')
        losses = []
        def ce(a, w, b): return F.cross_entropy(F.linear(a, w).float(), b, reduction='none')
        hh, yy = h.reshape(-1, self.c.d), y.reshape(-1)
        for start in range(0, len(yy), 1024):
            args = hh[start:start+1024], self.embed.weight, yy[start:start+1024]
            losses.append(checkpoint(ce, *args, use_reentrant=False) if self.training and torch.is_grad_enabled() else ce(*args))
        return torch.cat(losses).reshape(len(x), width).mean(1)
    def routed_nll(self, x, choices):
        if choices.shape != (len(x),) or ((choices < 0) | (choices >= 8)).any(): raise ValueError('Invalid routes')
        result = torch.zeros(len(x), device=x.device)
        for k, path in enumerate(ORDERS):
            ix = (choices == k).nonzero().flatten()
            if len(ix): result = result.index_copy(0, ix, self.nll(x[ix], path))
        return result

class Router(nn.Module):
    def __init__(self, d):
        super().__init__(); self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d,128),nn.SiLU(),nn.Linear(128,8))
    def forward(self, x): return self.net(x.float())

def router_loss(logits, costs, temperature=.02):
    # Full-information contextual bandit; actual counterfactual NLL labels.
    target = (-(costs.detach()-costs.detach().mean(-1,keepdim=True))/temperature).softmax(-1)
    return -(target * logits.log_softmax(-1)).sum(-1).mean()
