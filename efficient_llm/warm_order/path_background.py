"""Frozen, single-boundary INSID3-inspired diagnostic; no training or launcher.

A fitted direction is a candidate path-associated feature, never automatically
position bias. All transforms are fixed before the scored text is consumed.
"""
import torch
from torch.nn import functional as F

from .adaptation_analysis import _frozen_cpu, CANONICAL, SWAPPED
from .layer_dependence import _validate


@torch.no_grad()
def boundaries(model, tokens):
    _frozen_cpu(model)
    pos, seg = _validate(model, tokens)
    common = model.middle[0](model.stem(model.embed(tokens), pos, seg), pos, seg)
    a = model.middle[2](model.middle[1](common, pos, seg), pos, seg)
    b = model.middle[1](model.middle[2](common, pos, seg), pos, seg)
    return a, b


@torch.no_grad()
def suffix(model, tokens, h):
    _frozen_cpu(model)
    pos, seg = _validate(model, tokens)
    if h.shape != (*tokens.shape, model.c.d) or not torch.isfinite(h).all():
        raise ValueError('Invalid boundary state')
    return model.norm(model.coda(model.middle[3](h, pos, seg), pos, seg))


@torch.no_grad()
def score(model, tokens, h):
    """One NLL sum/count per sequence, identical prefix and tied readout."""
    hidden = suffix(model, tokens, h)[:, model.c.prefix-1:-1]
    targets = tokens[:, model.c.prefix:]
    logits_rows = hidden.reshape(-1, model.c.d)
    target_rows = targets.reshape(-1)
    pieces = [F.cross_entropy(F.linear(logits_rows[i:i+1024], model.embed.weight).float(),
                              target_rows[i:i+1024], reduction='none')
              for i in range(0, len(target_rows), 1024)]
    loss = torch.cat(pieces).reshape_as(targets)
    return loss.double().sum(1), torch.full((len(tokens),), targets.shape[1],
                                         dtype=torch.long, device=tokens.device)


def anchors(tokens, family, seed):
    """No null-meaning assumption; repeat/periodic fit, shuffle held out."""
    if tokens.ndim != 2 or tokens.dtype != torch.long or tokens.shape[1] < 2:
        raise ValueError('Expected actual token rows')
    if family == 'repeat': return tokens[:, :1].expand_as(tokens).clone()
    if family == 'periodic':
        return tokens[:, :2][:, torch.arange(tokens.shape[1], device=tokens.device)%2].clone()
    if family == 'shuffle':
        g = torch.Generator(device='cpu').manual_seed(seed)
        return torch.stack([row[torch.randperm(len(row), generator=g).to(row.device)] for row in tokens])
    raise ValueError('Unknown anchor family')


def observations(a, b, positions):
    if a.shape != b.shape or a.ndim != 3 or not positions or len(set(positions)) != len(positions):
        raise ValueError('Paired B,T,D states and distinct positions required')
    if any(type(p) is not int or p < 0 or p >= a.shape[1] for p in positions):
        raise ValueError('Invalid token position')
    return (b[:, positions]-a[:, positions]).reshape(-1, a.shape[-1]).detach().cpu().double()


def fit(deltas, rank):
    """Mean separately; centered right singular vectors, no test loss selection."""
    if deltas.ndim != 2 or len(deltas) < 2 or not torch.isfinite(deltas).all():
        raise ValueError('Finite multi-observation contrasts required')
    if type(rank) is not int or not 0 < rank < deltas.shape[1]:
        raise ValueError('Rank must be positive and smaller than hidden dimension')
    d = deltas.detach().cpu().double(); mean = d.mean(0)
    _, values, vh = torch.linalg.svd(d-mean, full_matrices=False)
    numerical_rank = int((values > max(float(values[0])*1e-8, 1e-12)).sum())
    actual = min(rank, numerical_rank)
    return dict(mean=mean, basis=vh[:actual].T.contiguous(), singular_values=values,
                requested_rank=rank, actual_rank=actual, observations=len(d))


def random_basis(dim, rank, seed):
    if not 0 <= rank < dim: raise ValueError('Invalid control rank')
    g = torch.Generator(device='cpu').manual_seed(seed)
    return torch.linalg.qr(torch.randn(dim, rank, generator=g, dtype=torch.float64), mode='reduced')[0]


def _project(x, basis):
    u = basis.to(device=x.device, dtype=x.dtype)
    if u.ndim != 2 or u.shape[0] != x.shape[-1] or not torch.isfinite(u).all():
        raise ValueError('Invalid fitted basis')
    if not torch.allclose(u.T@u, torch.eye(u.shape[1], device=u.device, dtype=u.dtype), atol=2e-5, rtol=2e-5):
        raise ValueError('Orthonormal basis required')
    return (x@u)@u.T


def transform(h, fitted, mode, control=None):
    """Fixed transforms; matched controls use recipient only, no native oracle."""
    if not torch.isfinite(h).all(): raise ValueError('Nonfinite state')
    mean = fitted['mean'].to(device=h.device, dtype=h.dtype)
    if mean.shape != (h.shape[-1],): raise ValueError('Mean shape mismatch')
    if mode == 'identity': return h
    if mode == 'minus_mean': return h-mean
    if mode == 'plus_mean': return h+mean
    removed = _project(h, fitted['basis'])
    if mode == 'project': return h-removed
    if mode == 'project_norm_restored':
        result = h-removed; oldnorm = h.norm(dim=-1, keepdim=True); newnorm = result.norm(dim=-1, keepdim=True)
        if ((oldnorm>1e-12)&(newnorm<=1e-12)).any(): raise ValueError('Projection destroys nonzero state')
        return result*torch.where(newnorm>1e-12, oldnorm/newnorm.clamp_min(1e-12), torch.ones_like(newnorm))
    if mode in ('random_project','random_energy_matched'):
        if control is None or control.shape != fitted['basis'].shape: raise ValueError('Matched control basis required')
        r = _project(h, control)
        if mode == 'random_project': return h-r
        source = r.norm(dim=-1,keepdim=True); target = removed.norm(dim=-1,keepdim=True)
        if ((target>1e-12)&(source<=1e-12)).any(): raise ValueError('Degenerate random energy control')
        return h-r*(target/source.clamp_min(1e-12))
    raise ValueError('Unknown transform')


def transfer(deltas, fitted):
    d = deltas.detach().cpu().double(); projection = _project(d, fitted['basis'])
    total = float(d.square().sum())
    return dict(n=len(d),raw_difference_energy=total,
                projected_energy_fraction=float(projection.square().sum())/total if total>0 else None,
                original_mean_square=float(d.square().sum(1).mean()),
                after_mean_correction_mean_square=float((d-fitted['mean']).square().sum(1).mean()),
                interpretation='Descriptive transfer only, not causal share or semantic purity')


def overlap(left, right):
    u,v=left['basis'],right['basis'];rank=min(u.shape[1],v.shape[1])
    return float((u.T@v).square().sum()/rank) if rank else None
