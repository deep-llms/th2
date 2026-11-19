"""Authenticated phase entry; preserve parent Adam state, disable mixture floor."""
import json
import random
from pathlib import Path

import numpy as np
import torch

from .data import sha
from .model import Config
from .soft_route import Model as SoftModel, ARMS
from .soft_exploration import Model as ExposureModel


def optimizer_for(model, body_lr, router_lr):
    router_names = {'depth_logits', 'route_norm.weight', 'route_norm.bias',
                    'route_linear.weight'}
    body = [p for n, p in model.named_parameters()
            if p.requires_grad and n not in router_names]
    router = [p for n, p in model.named_parameters()
              if p.requires_grad and n in router_names]
    groups = [dict(params=body, peak=body_lr)]
    if router:
        groups.append(dict(params=router, peak=router_lr))
    return torch.optim.AdamW(groups, lr=body_lr, betas=(.9, .95), weight_decay=.1)


def load_parent(directory, hashes, name, *, expected_steps=2000, device='cpu'):
    """Only load trusted, hash-pinned terminal files. Does not change any file.

    Returns a model, optimizer, and original state. RNG restoration is explicit
    so callers can finish allocations before restoring the exact saved streams.
    The future phase trainer must keep the inherited group LRs constant.
    """
    directory = Path(directory)
    files = ('fullstate.pt', 'weights.pt', 'complete.json', 'recipe.json')
    if set(hashes) != set(files):
        raise ValueError('All four parent identities required')
    for f in files:
        if sha(directory / f) != hashes[f]:
            raise ValueError('Parent file hash mismatch: ' + f)
    pieces = name.split('-s')
    if len(pieces) != 2 or pieces[0] not in ('C', 'G', 'S', 'GF', 'SF') or pieces[1] not in ('0', '1'):
        raise ValueError('Unregistered parent name')
    arm, seed = pieces[0][0], int(pieces[1])
    exposure = pieces[0].endswith('F')
    done = json.loads((directory / 'complete.json').read_text())
    recipe = json.loads((directory / 'recipe.json').read_text())
    c = recipe['config']
    if (done['step'] != expected_steps or c['steps'] != expected_steps or
            done['seed'] != seed or c['seed'] != seed or
            done['arm'] != arm or c['arm'] != arm or
            done['fullstate_sha256'] != hashes['fullstate.pt'] or
            done['weights_sha256'] != hashes['weights.pt']):
        raise ValueError('Parent terminal identity mismatch')
    if exposure != ('exploration_schedule' in c):
        raise ValueError('Parent schedule/schema mismatch')
    state = torch.load(directory / 'fullstate.pt', map_location='cpu', weights_only=False)
    weights = torch.load(directory / 'weights.pt', map_location='cpu', weights_only=False)
    if state['recipe'] != recipe or state['step'] != expected_steps:
        raise ValueError('Parent fullstate recipe mismatch')
    if (weights['config'] != c['model'] or weights['arm'] != arm or
            weights['seed'] != seed or weights['canonical_probability'] != c['canonical_probability']):
        raise ValueError('Parent weights metadata mismatch')
    if state['model'].keys() != weights['model'].keys() or any(
            not torch.equal(v, weights['model'][k]) for k, v in state['model'].items()):
        raise ValueError('Parent weights/fullstate tensors differ')
    cls = ExposureModel if exposure else SoftModel
    model = cls(Config(**c['model']), ARMS[arm], c['canonical_probability'])
    model.load_state_dict(state['model'], strict=True)
    if exposure:
        if int(model.exploration_update) != expected_steps - 1:
            raise ValueError('Parent exposure position mismatch')
        model.disable_exploration()
    model.to(device)
    optimizer = optimizer_for(model, c['lr'], c['router_lr'])
    groups = state['optimizer']['param_groups']
    peaks = [c['lr']] + ([c['router_lr']] if arm != 'C' else [])
    if len(groups) != len(peaks):
        raise ValueError('Parent optimizer group count mismatch')
    for g, peak in zip(groups, peaks):
        if (g['peak'] != peak or abs(g['lr'] - .1 * peak) > 1e-12 or
                tuple(g['betas']) != (.9, .95) or g['weight_decay'] != .1):
            raise ValueError('Parent terminal optimizer settings mismatch')
    parameter_ids = [k for g in groups for k in g['params']]
    moments = state['optimizer']['state']
    if len(set(parameter_ids)) != len(parameter_ids) or set(moments) != set(parameter_ids):
        raise ValueError('Parent Adam moments missing or duplicated')
    for values in moments.values():
        if (not {'step', 'exp_avg', 'exp_avg_sq'} <= set(values) or
                int(values['step']) != expected_steps):
            raise ValueError('Parent Adam update counter mismatch')
    optimizer.load_state_dict(state['optimizer'])
    return model, optimizer, state


def restore_rng(state, device='cpu'):
    """Call after model/optimizer allocations, immediately before training."""
    if str(device).startswith('cuda'):
        if len(state['cuda_rng']) != torch.cuda.device_count():
            raise ValueError('CUDA RNG device count mismatch')
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    torch.set_rng_state(state['torch_rng'])
    np.random.set_state(state['numpy_rng'])
    random.setstate(state['python_rng'])
