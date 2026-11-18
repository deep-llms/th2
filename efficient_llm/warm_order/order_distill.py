"""Exact full-vocabulary teacher KL for two equal-depth causal orders."""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

IDENTITY = (0, 1, 2, 3)
SWAP = (0, 1, 3, 2)
ARMS = ('C2', 'K2', 'P2', 'PK2')


def chunked_objective(hidden, weight, teacher_hidden, teacher_weight,
                      targets, chunk=1024):
    """Per-token CE and forward KL, reduced over all scored tokens exactly."""
    if chunk < 1 or hidden.shape != teacher_hidden.shape:
        raise ValueError('Invalid chunk or teacher shape')
    h = hidden.reshape(-1, hidden.shape[-1])
    t = teacher_hidden.detach().reshape_as(h)
    y = targets.reshape(-1)
    if len(y) != len(h) or len(y) == 0:
        raise ValueError('Token alignment mismatch')
    tw = teacher_weight.detach()

    def piece(a, w, b, frozen_w, labels):
        logp = F.log_softmax(F.linear(a, w).float(), dim=-1)
        with torch.no_grad():
            logq = F.log_softmax(F.linear(b, frozen_w).float(), dim=-1)
            q = logq.exp()
        ce = F.nll_loss(logp, labels, reduction='sum')
        kl = (q * (logq - logp)).sum()
        return torch.stack((ce, kl))

    pieces = []
    for start in range(0, len(y), chunk):
        args = h[start:start+chunk], weight, t[start:start+chunk], tw, y[start:start+chunk]
        pieces.append(checkpoint(piece, *args, use_reentrant=False)
                      if torch.is_grad_enabled() else piece(*args))
    return torch.stack(pieces).sum(0) / len(y)


def objective(student, teacher, x, arm, chunk=1024):
    if arm not in ARMS or student.c != teacher.c:
        raise ValueError('Invalid arm or model mismatch')
    if any(p.requires_grad for p in teacher.parameters()):
        raise ValueError('Teacher must be explicitly frozen')
    prefix = student.c.prefix
    with torch.no_grad():
        target_hidden = teacher.hidden(x, IDENTITY)[:, prefix-1:-1]
    paths = (IDENTITY, SWAP if arm in ('P2', 'PK2') else IDENTITY)
    values = []
    for path in paths:
        hidden = student.hidden(x, path)[:, prefix-1:-1]
        values.append(chunked_objective(hidden, student.embed.weight,
                      target_hidden, teacher.embed.weight, x[:, prefix:], chunk))
    ce, kl = torch.stack(values).mean(0).unbind()
    return ce + (0.5 if arm in ('K2', 'PK2') else 0.) * kl, ce, kl
