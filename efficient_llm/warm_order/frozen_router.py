"""Router-only learning through an immutable dense backbone."""
import torch
from torch import nn
from .soft_route import Model as SoftModel
from .model import Model as Dense


class Model(SoftModel):
    def __init__(self, config, arm='M', canonical_probability=.99, router_width=128):
        if arm not in ('G', 'L', 'M') or router_width < 1:
            raise ValueError('Invalid frozen-router architecture')
        super().__init__(config, 'global' if arm == 'G' else 'dynamic', canonical_probability)
        self.arm = arm
        self.depth_heads = nn.ModuleList()
        if arm == 'M':
            for _ in range(4):
                head = nn.Sequential(nn.LayerNorm(config.d), nn.Linear(config.d, router_width),
                                     nn.SiLU(), nn.Linear(router_width, 4))
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
                self.depth_heads.append(head)
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(self.is_router(name))

    def is_router(self, name):
        return (name == 'depth_logits' or
                self.arm == 'L' and name.startswith(('route_norm.', 'route_linear.')) or
                self.arm == 'M' and name.startswith('depth_heads.'))

    def gates(self, h, depth):
        logits = self.depth_logits[depth].expand(*h.shape[:2], 4)
        if self.arm == 'L':
            features = h.to(self.route_linear.weight.dtype)
            logits = logits + self.route_linear(self.route_norm(features))
        elif self.arm == 'M':
            head = self.depth_heads[depth]
            logits = logits + head(h.to(head[1].weight.dtype))
        return logits.softmax(-1)

    def load_dense_backbone(self, state):
        # Build the schema without relying on prefixes to decide what is a body tensor.
        # Preserve RNG because constructing a schema model initializes parameters.
        with torch.random.fork_rng(devices=[]):
            keys = set(Dense(self.c).state_dict())
        if set(state) != keys:
            raise ValueError('Dense backbone schema mismatch')
        own = self.state_dict()
        for name, value in state.items():
            if own[name].shape != value.shape or own[name].dtype != value.dtype:
                raise ValueError('Dense backbone shape/dtype mismatch: ' + name)
        with torch.no_grad():
            for name, value in state.items():
                own[name].copy_(value)

    def router_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def frozen_state(self):
        return {n: p.detach().clone() for n, p in self.named_parameters() if not p.requires_grad}

    def assert_frozen(self, before):
        current = dict(self.named_parameters())
        expected = {n for n, p in current.items() if not p.requires_grad}
        if set(before) != expected or any(not torch.equal(current[n].detach().to(before[n].device), before[n]) for n in expected):
            raise ValueError('Frozen tensor changed')
        if any(current[n].grad is not None for n in expected):
            raise ValueError('Unexpected body gradient')


def load_initial(c):
    """Authenticate completed recovery C weights and sidecars; fresh router only."""
    import json
    from pathlib import Path
    from .data import sha
    from .model import Config
    parent = Path(c['parent'])
    required = {'weights.pt', 'fullstate.pt', 'complete.json', 'recipe.json'}
    if set(c['parent_hashes']) != required:
        raise ValueError('Four parent pins required')
    for name, digest in c['parent_hashes'].items():
        if sha(parent/name) != digest:
            raise ValueError('Parent identity mismatch: ' + name)
    done = json.loads((parent/'complete.json').read_text())
    recipe = json.loads((parent/'recipe.json').read_text())
    pc = recipe['config']
    if (done['name'] != f"C-s{c['seed']}" or done['arm'] != 'C' or
            done['seed'] != c['seed'] or done['step'] != c['parent_steps'] or
            pc['name'] != done['name'] or pc['steps'] != done['step'] or
            pc['model'] != c['model'] or
            done['weights_sha256'] != c['parent_hashes']['weights.pt'] or
            done['fullstate_sha256'] != c['parent_hashes']['fullstate.pt']):
        raise ValueError('Recovery C parent metadata mismatch')
    weights = torch.load(parent/'weights.pt', map_location='cpu', weights_only=False)
    full = torch.load(parent/'fullstate.pt', map_location='cpu', weights_only=False)
    if (full['recipe'] != recipe or full['step'] != done['step'] or
            full['model'].keys() != weights['model'].keys() or
            any(not torch.equal(v, weights['model'][k]) for k, v in full['model'].items())):
        raise ValueError('Parent weights/fullstate mismatch')
    del full
    if weights['config'] != c['model'] or weights['arm'] != 'C' or weights['seed'] != c['seed']:
        raise ValueError('Wrong parent weights')
    # The recovery parent includes unused routing parameters. Authenticate and
    # load its complete schema before extracting only dense backbone tensors.
    old = SoftModel(Config(**c['model']), 'dense', weights['canonical_probability'])
    old.load_state_dict(weights['model'], strict=True)
    with torch.random.fork_rng(devices=[]):
        keys = set(Dense(old.c).state_dict())
    m = Model(old.c, c['arm'], c['canonical_probability'], c['router_width'])
    m.load_dense_backbone({k: old.state_dict()[k] for k in keys})
    return m
