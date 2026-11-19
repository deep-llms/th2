"""Causal mixtures of complete AB/BA compositions; no sparse-speed claim."""
import math
import torch
from torch import nn
from .model import Model as Dense
from .soft_route import combine


class Model(Dense):
    def __init__(self, config, arm='M', canonical_probability=.99, router_width=128):
        if arm not in ('C', 'G', 'M') or not 0 < canonical_probability < 1 or router_width < 1:
            raise ValueError('Invalid pair-order configuration')
        super().__init__(config)
        self.arm = arm
        self.pair_logits = nn.Parameter(torch.tensor([
            [math.log(canonical_probability), math.log(1-canonical_probability)]
        ] * 2), requires_grad=arm != 'C')
        self.pair_heads = nn.ModuleList()
        if arm == 'M':
            for _ in range(2):
                head = nn.Sequential(nn.LayerNorm(config.d), nn.Linear(config.d, router_width),
                                     nn.SiLU(), nn.Linear(router_width, 2))
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
                self.pair_heads.append(head)

    @staticmethod
    def is_router(name):
        return name == 'pair_logits' or name.startswith('pair_heads.')

    def load_dense(self, state):
        # Validate fully before copying any tensor; schema construction preserves RNG.
        with torch.random.fork_rng(devices=[]):
            keys = set(Dense(self.c).state_dict())
        own = self.state_dict()
        if set(state) != keys:
            raise ValueError('Dense parent schema mismatch')
        if any(own[k].shape != state[k].shape or own[k].dtype != state[k].dtype for k in keys):
            raise ValueError('Dense parent shape/dtype mismatch')
        with torch.no_grad():
            for key in keys:
                own[key].copy_(state[key])

    def gates(self, h, pair):
        if self.arm == 'C':
            return h.new_full((*h.shape[:2], 2), .5, dtype=self.pair_logits.dtype)
        logits = self.pair_logits[pair].expand(*h.shape[:2], 2)
        if self.arm == 'M':
            head = self.pair_heads[pair]
            logits = logits + head(h.to(head[1].weight.dtype))
        return logits.softmax(-1)

    def hidden_and_gates(self, x, hard=False):
        if x.ndim != 2 or not 0 < x.shape[1] <= self.c.length:
            raise ValueError('Invalid sequence shape')
        pos = torch.arange(x.shape[1], device=x.device).expand_as(x)
        seg = torch.zeros_like(x)
        h = self.stem(self.embed(x), pos, seg)
        probabilities = []
        for pair in range(2):
            a, b = self.middle[2*pair:2*pair+2]
            p = self.gates(h, pair)
            canonical = b(a(h, pos, seg), pos, seg)
            # C repeats the computation rather than detaching or reusing the first branch.
            alternate = (b(a(h, pos, seg), pos, seg) if self.arm == 'C'
                         else a(b(h, pos, seg), pos, seg))
            h = combine(torch.stack((canonical, alternate), dim=-2), p, hard)
            probabilities.append(p)
        self.last_routes = [p.detach().argmax(-1) for p in probabilities]
        return self.norm(self.coda(h, pos, seg)), probabilities

    def hidden(self, x, hard=False):
        return self.hidden_and_gates(x, hard)[0]

    def nll(self, x, hard=False):
        return super().nll(x, hard)


def load_initial(config):
    """Reuse authenticated recovery-C loading, then copy only the dense schema."""
    from .frozen_router import load_initial as authenticate_parent
    authenticated = authenticate_parent({**config, 'arm': 'G'})
    with torch.random.fork_rng(devices=[]):
        keys = set(Dense(authenticated.c).state_dict())
    model = Model(authenticated.c, config['arm'], config['canonical_probability'], config['router_width'])
    state = authenticated.state_dict()
    model.load_dense({key: state[key] for key in keys})
    return model
