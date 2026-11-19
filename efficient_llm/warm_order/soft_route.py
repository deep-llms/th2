"""Continuous causal whole-block mixtures; dense computation, no sparse-speed claim."""
import math
import json
from pathlib import Path
import torch
from torch import nn
from .model import Model as Dense


def combine(outputs, probabilities, hard=False):
    """outputs[B,T,K,D], probabilities[B,T,K]; hard projection has no surrogate."""
    if outputs.shape[:-1] != probabilities.shape:
        raise ValueError('Incompatible mixture shapes')
    if hard:
        choice = probabilities.argmax(-1)
        return outputs.gather(-2, choice[..., None, None].expand(*choice.shape, 1, outputs.shape[-1])).squeeze(-2)
    # Accumulate BF16/FP16 branches in FP32; retain double precision for audit.
    dtype = torch.float32 if outputs.dtype in (torch.float16, torch.bfloat16) else outputs.dtype
    return (outputs.to(dtype) * probabilities.to(dtype)[..., None]).sum(-2).to(outputs.dtype)


class Model(Dense):
    def __init__(self, config, mode='dynamic', canonical_probability=.999):
        if mode not in ('dense', 'global', 'dynamic') or not 0 < canonical_probability < 1:
            raise ValueError('Invalid soft-routing configuration')
        super().__init__(config)
        self.mode = mode
        prior = torch.full((4, 4), math.log((1-canonical_probability)/3))
        prior.fill_diagonal_(math.log(canonical_probability))
        self.depth_logits = nn.Parameter(prior, requires_grad=mode != 'dense')
        self.route_norm = nn.LayerNorm(config.d)
        self.route_linear = nn.Linear(config.d, 4, bias=False)
        nn.init.zeros_(self.route_linear.weight)
        for p in (*self.route_norm.parameters(), *self.route_linear.parameters()):
            p.requires_grad_(mode == 'dynamic')

    def load_dense(self, state):
        missing, extra = self.load_state_dict(state, strict=False)
        if set(missing) != {'depth_logits', 'route_norm.weight', 'route_norm.bias', 'route_linear.weight'} or extra:
            raise ValueError('Dense parent schema mismatch')

    def gates(self, h, depth):
        logits = self.depth_logits[depth].expand(*h.shape[:2], 4)
        if self.mode == 'dynamic':
            features = h.to(self.route_linear.weight.dtype)
            logits = logits + self.route_linear(self.route_norm(features))
        return logits.softmax(-1)

    def hidden_and_gates(self, x, hard=False):
        if x.ndim != 2 or not 0 < x.shape[1] <= self.c.length:
            raise ValueError('Invalid sequence shape')
        pos = torch.arange(x.shape[1], device=x.device).expand_as(x)
        seg = torch.zeros_like(x)
        h = self.stem(self.embed(x), pos, seg)
        probabilities = []
        for depth in range(4):
            if self.mode == 'dense':
                h = self.middle[depth](h, pos, seg)
            else:
                p = self.gates(h, depth)
                outputs = torch.stack([block(h, pos, seg) for block in self.middle], -2)
                h = combine(outputs, p, hard)
                probabilities.append(p)
        self.last_routes = [p.detach().argmax(-1) for p in probabilities]
        return self.norm(self.coda(h, pos, seg)), probabilities

    def hidden(self, x, hard=False):
        return self.hidden_and_gates(x, hard)[0]

    def nll(self, x, hard=False):
        return super().nll(x, hard)


ARMS = {'C': 'dense', 'G': 'global', 'S': 'dynamic'}

def load_initial(config):
    from .model import Config
    from .data import sha
    weights = Path(config['initial_weights'])
    metadata = config['parent_metadata']
    required = {str(weights.parent/'complete.json'), str(weights.parent/'recipe.json')}
    if set(metadata) != required or any(sha(p) != h for p,h in metadata.items()):
        raise ValueError('Parent metadata identity mismatch')
    done = json.loads((weights.parent/'complete.json').read_text())
    recipe = json.loads((weights.parent/'recipe.json').read_text())['config']
    if (done['arm'] != 'D' or recipe['arm'] != 'D' or
        done['seed'] != config['seed'] or recipe['seed'] != config['seed'] or
        done['step'] != config['parent_steps'] or recipe['steps'] != config['parent_steps'] or
        recipe['model'] != config['model'] or sha(weights) != done['weights_sha256']):
        raise ValueError('Dense parent metadata mismatch')
    s = torch.load(weights, map_location='cpu', weights_only=False)
    if set(s) != {'config','model'} or s['config'] != config['model']:
        raise ValueError('Unexpected Horizon weights schema')
    m = Model(Config(**config['model']), ARMS[config['arm']], config['canonical_probability'])
    m.load_dense(s['model'])
    return m
