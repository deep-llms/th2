"""Descriptive, streaming activation geometry; never a causal attribution.

Observation = a registered sequence/token position. Feature centering is across
observations. Centered moments merge stably without storing all activations.
"""
import numpy as np
import torch

from .adaptation_analysis import native_trace, CANONICAL, SWAPPED


class GeometryMoments:
    def __init__(self, width):
        if type(width) is not int or width <= 0: raise ValueError('Positive feature width required')
        self.width, self.n = width, 0
        self.mx = torch.zeros(width, dtype=torch.float64)
        self.my = self.mx.clone()
        self.xx = torch.zeros(width, width, dtype=torch.float64)
        self.yy = self.xx.clone(); self.xy = self.xx.clone()
        self.x2 = self.y2 = self.delta2 = self.cos = 0.
        self.cos_count = 0

    def arrays(self):
        """Lossless sufficient statistics for local/node-side numeric replay."""
        result = {k: getattr(self, k).numpy().copy() for k in ('mx', 'my', 'xx', 'yy', 'xy')}
        result['counts'] = np.array([self.width, self.n, self.cos_count], dtype=np.int64)
        result['sums'] = np.array([self.x2, self.y2, self.delta2, self.cos], dtype=np.float64)
        return result

    @classmethod
    def from_arrays(cls, arrays):
        if set(arrays) != {'mx', 'my', 'xx', 'yy', 'xy', 'counts', 'sums'}:
            raise ValueError('Unexpected moment members')
        counts, sums = arrays['counts'], arrays['sums']
        if counts.shape != (3,) or counts.dtype != np.int64 or sums.shape != (4,) or sums.dtype != np.float64:
            raise ValueError('Invalid moment counts/sums schema')
        width, n, valid = map(int, counts)
        if width <= 0 or n <= 0 or not 0 <= valid <= n:
            raise ValueError('Invalid moment counts')
        if not np.isfinite(sums).all() or (sums[:3] < 0).any() or abs(sums[3]) > valid+1e-8:
            raise ValueError('Invalid moment sums')
        for key in ('mx', 'my', 'xx', 'yy', 'xy'):
            value = arrays[key]; shape = (width,) if key in ('mx', 'my') else (width, width)
            if value.shape != shape or value.dtype != np.float64 or not np.isfinite(value).all():
                raise ValueError('Invalid moment matrix/vector')
        result = cls(width)
        for key in ('mx', 'my', 'xx', 'yy', 'xy'):
            setattr(result, key, torch.from_numpy(arrays[key].copy()))
        result.n, result.cos_count = n, valid
        result.x2, result.y2, result.delta2, result.cos = map(float, sums)
        return result

    def update(self, x, y):
        if (x.shape != y.shape or x.ndim != 2 or x.shape[1] != self.width or
                x.device.type != 'cpu' or y.device.type != 'cpu' or not len(x)):
            raise ValueError('Matched nonempty CPU observations-by-features required')
        x, y = x.detach().double(), y.detach().double()
        if not torch.isfinite(x).all() or not torch.isfinite(y).all(): raise ValueError('Nonfinite activation')
        n = len(x); mx, my = x.mean(0), y.mean(0)
        xc, yc = x-mx, y-my
        dx, dy = mx-self.mx, my-self.my
        factor = self.n*n/(self.n+n)
        self.xx += xc.T@xc + factor*torch.outer(dx, dx)
        self.yy += yc.T@yc + factor*torch.outer(dy, dy)
        self.xy += xc.T@yc + factor*torch.outer(dx, dy)
        self.mx += dx*(n/(self.n+n)); self.my += dy*(n/(self.n+n))
        self.n += n
        nx, ny = x.norm(dim=1), y.norm(dim=1)
        valid = (nx > 0) & (ny > 0)
        self.cos += float(((x[valid]*y[valid]).sum(1)/(nx[valid]*ny[valid])).sum())
        self.cos_count += int(valid.sum())
        self.x2 += float(x.square().sum()); self.y2 += float(y.square().sum())
        self.delta2 += float((x-y).square().sum())

    def report(self):
        if not self.n: raise ValueError('No observations')
        denominator = float(self.xx.norm()*self.yy.norm())
        result = dict(observations=self.n, features=self.width,
            rms_x=(self.x2/(self.n*self.width))**.5, rms_y=(self.y2/(self.n*self.width))**.5,
            mean_squared_distance=self.delta2/self.n,
            mean_cosine=self.cos/self.cos_count if self.cos_count else None,
            cosine_valid_count=self.cos_count,
            centered_linear_cka=float(self.xy.square().sum())/denominator if denominator > 0 else None,
            cka_defined=denominator > 0,
            mean_shift_squared=float((self.mx-self.my).square().sum()),
            interpretation='Descriptive geometry only; no task importance or causal percentage')
        for key, matrix in [('x', self.xx), ('y', self.yy)]:
            eigenvalues = torch.linalg.eigvalsh(matrix).flip(0)
            total = float(matrix.trace())
            result[key+'_scatter_top8_eigenvalues'] = eigenvalues[:8].tolist()
            result[key+'_scatter_trace'] = total
            result[key+'_top8_variance_fraction'] = float(eigenvalues[:8].sum())/total if total > 0 else None
        return result


@torch.no_grad()
def aligned_observations(model_x, model_y, tokens, positions, *, path_x=CANONICAL, path_y=SWAPPED):
    """Small selected observations, with explicit slot AND block alignment.

    Selection positions are fixed by caller before viewing outcomes. Full native
    traces exist only for this batch. No pooling across alignment definitions.
    """
    if (not isinstance(positions, tuple) or not positions or len(set(positions)) != len(positions) or
            any(type(p) is not int or p < 0 or p >= tokens.shape[1] for p in positions)):
        raise ValueError('Explicit unique valid token positions required')
    _, left = native_trace(model_x, tokens, path_x)
    _, right = native_trace(model_y, tokens, path_y)
    by_name = {r['block']: r for r in right}
    output = {}
    for alignment in ('slot', 'block'):
        for i, a in enumerate(left):
            b = right[i] if alignment=='slot' else by_name[a['block']]
            for field in ('input', 'attention_update', 'ffn_update', 'output'):
                name = f'{alignment}/{i}/{field}'
                output[name] = dict(x=a[field][:, positions].reshape(-1, model_x.c.d).clone(),
                                    y=b[field][:, positions].reshape(-1, model_y.c.d).clone(),
                                    left_block=a['block'], right_block=b['block'],
                                    left_slot=a['execution_slot'], right_slot=b['execution_slot'])
    return output
