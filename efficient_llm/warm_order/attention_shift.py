"""One constant attention-output correction; the original body stays frozen."""
from contextlib import contextmanager
import hashlib
import json

import torch

PATHS = {'C': [0, 1, 2, 3], 'S': [0, 2, 1, 3]}


def freeze(model):
    model.requires_grad_(False)
    for p in model.parameters():
        p.grad = None
    return model


def fingerprint(model):
    """Cover parameters AND nonpersistent buffers, without retaining a clone."""
    h = hashlib.sha256()
    for kind, values in [('parameter', model.named_parameters()), ('buffer', model.named_buffers())]:
        for name, value in values:
            h.update(json.dumps([kind, name, str(value.dtype), list(value.shape)]).encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@contextmanager
def inject(model, bias):
    if (bias.shape != (model.c.d,) or bias.device != model.embed.weight.device or
            bias.dtype != torch.float32 or not torch.isfinite(bias).all()):
        raise ValueError('Finite FP32 correction on model device required')
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise ValueError('Body must be frozen with no parameter gradients')
    attention = model.middle[2].self_attn
    if attention._forward_hooks:
        raise ValueError('Unexpected attention hook; refuse stacking')
    calls = []

    def add(_module, _args, output):
        calls.append(1)
        return output + bias.to(dtype=output.dtype)

    handle = attention.register_forward_hook(add)
    try:
        yield
        if len(calls) != 1:
            raise ValueError('Expected exactly one target attention invocation')
    finally:
        handle.remove()


def shifted(model, x, bias, branch, *, hidden=False):
    if branch not in PATHS:
        raise ValueError('Only registered canonical/swap paths supported')
    with inject(model, bias):
        return model.hidden(x, PATHS[branch]) if hidden else model.nll(x, PATHS[branch])
