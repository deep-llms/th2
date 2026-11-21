"""Fixed V/O update diagnostics; no runner, fitting of weights, or admission.

The per-input donor delta is required: these patches are not standalone
translators. Calibration/confirmation identity belongs to the bound driver.
"""
from dataclasses import asdict
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from .adaptation_analysis import _frozen_cpu
from .adaptation_components import _attention_parts
from .adaptation_geometry import GeometryMoments
from .layer_dependence import _validate
from .path_background import _project, random_basis

CONTROL_SEEDS = (20261121, 20261122, 20261123)


class DeltaMoments:
    """Reuse stable centered moment merges, retaining no observation matrix."""
    def __init__(self, width):
        self.moments = GeometryMoments(width)

    def update(self, delta):
        self.moments.update(torch.zeros_like(delta), delta)

    def arrays(self):
        return self.moments.arrays()

    @classmethod
    def from_arrays(cls, arrays):
        moments = GeometryMoments.from_arrays(arrays)
        if (torch.count_nonzero(moments.mx) or torch.count_nonzero(moments.xx)
                or torch.count_nonzero(moments.xy) or moments.x2 != 0):
            raise ValueError('Expected zero-reference delta moments')
        out = cls(moments.width)
        out.moments = moments
        return out

    def fit(self, rank=8):
        m = self.moments
        if m.n < 2 or type(rank) is not int or not 0 < rank < m.width:
            raise ValueError('Multiple observations and valid fixed rank required')
        matrix = (m.yy + m.yy.T) / 2
        values, vectors = torch.linalg.eigh(matrix)
        values, vectors = values.flip(0), vectors.flip(1)
        threshold = max(float(values[0]) * 1e-10, 1e-12)
        if float(values[-1]) < -threshold:
            raise ValueError('Invalid negative scatter eigenvalue')
        actual = min(rank, int((values > threshold).sum()))
        return dict(mean=m.my.clone(), basis=vectors[:, :actual].contiguous(),
                    eigenvalues=values, requested_rank=rank, actual_rank=actual,
                    observations=m.n, numerical_threshold=threshold,
                    boundary_eigengap=float(values[actual-1]-values[actual]) if actual else None)


def _match_norm(vector, target):
    source = vector.norm(dim=-1, keepdim=True)
    if not torch.isfinite(target).all() or (target < 0).any():
        raise ValueError('Invalid target norm')
    if ((source <= 1e-12) & (target > 1e-12)).any():
        raise ValueError('Degenerate norm-matching direction')
    scale = torch.where(source > 1e-12, target/source.clamp_min(1e-12), torch.zeros_like(source))
    return vector * scale


@torch.no_grad()
def update_cases(u0, donor, fitted):
    """Apply frozen calibration basis independently at every causal token."""
    if (u0.shape != donor.shape or u0.ndim != 3 or u0.device != donor.device
            or u0.dtype != donor.dtype or not torch.isfinite(u0).all()
            or not torch.isfinite(donor).all()):
        raise ValueError('Finite matched B,T,D updates required')
    mean = fitted['mean'].to(u0)
    basis = fitted['basis']
    if mean.shape != (u0.shape[-1],) or not torch.isfinite(mean).all():
        raise ValueError('Invalid calibration mean')
    delta = donor-u0
    projected = _project(delta, basis)
    centered = _project(delta-mean, basis)
    patched = u0+projected
    cases = dict(base=u0, full=donor, mean=u0+mean, projected=patched,
                 complement=u0+delta-projected, centered=u0+centered,
                 affine=u0+mean+centered, reverse=u0-projected,
                 radial=_match_norm(u0, patched.norm(dim=-1, keepdim=True)),
                 norm_restored=_match_norm(patched, u0.norm(dim=-1, keepdim=True)))
    for seed in CONTROL_SEEDS:
        control = random_basis(u0.shape[-1], basis.shape[1], seed)
        random_delta = _project(delta, control)
        cases[f'random/{seed}'] = u0+random_delta
        cases[f'random_energy/{seed}'] = u0+_match_norm(random_delta, projected.norm(dim=-1, keepdim=True))
    if any(not torch.isfinite(x).all() for x in cases.values()):
        raise ValueError('Nonfinite directional patch')
    return cases


@torch.no_grad()
def vo_boundary(parent, adapted, tokens):
    """Parent H0/QK/norms; donor V/O only, original SDPA arithmetic."""
    for model in (parent, adapted):
        _frozen_cpu(model)
        _validate(model, tokens)
    if asdict(parent.c) != asdict(adapted.c):
        raise ValueError('Matching configurations required')
    pos, seg = _validate(parent, tokens)
    h = parent.middle[0](parent.stem(parent.embed(tokens), pos, seg), pos, seg)
    block = parent.middle[2]
    attention, value = block.self_attn, adapted.middle[2].self_attn
    x = block.input_layernorm(h)
    u0 = attention(x, pos, seg)
    mixed = SimpleNamespace(c=attention.c, inv=attention.inv,
                            q_proj=attention.q_proj, k_proj=attention.k_proj,
                            v_proj=value.v_proj, q_norm=attention.q_norm, k_norm=attention.k_norm)
    parts = _attention_parts(mixed, x)
    mask = (seg[:, :, None] == seg[:, None, :]) & torch.ones(
        h.shape[1], h.shape[1], dtype=torch.bool, device=h.device).tril()
    joined = F.scaled_dot_product_attention(parts['q'], parts['k'], parts['value'], attn_mask=mask[:, None])
    donor = value.o_proj(joined.transpose(1, 2).reshape(h.shape))
    return dict(h=h, base=u0, donor=donor, pos=pos, segments=seg)


@torch.no_grad()
def finish_update(parent, boundary, update):
    h, pos, seg = boundary['h'], boundary['pos'], boundary['segments']
    if update.shape != h.shape or not torch.isfinite(update).all():
        raise ValueError('Finite matching update required')
    z = h+update
    block = parent.middle[2]
    y = z+block.mlp(block.post_attention_layernorm(z))
    for index in (1, 3):
        y = parent.middle[index](y, pos, seg)
    return parent.norm(parent.coda(y, pos, seg))
