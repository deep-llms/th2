"""Block-routed P4/P6 estimators and the depth-only P5 stream."""
import torch
from torch import nn
from torch.nn import functional as F
from .model import normalize_code


def gated_prediction(prediction, alpha, source):
    return alpha[0]*normalize_code(prediction).to(source.dtype)


def cosine_loss(prediction, target, valid):
    with torch.autocast(prediction.device.type, enabled=False):
        cosine = (F.cosine_similarity(prediction.float(),target.detach().float(),dim=-1,eps=1e-6)*valid).sum(-1)
        return cosine[:,None], (valid.sum(-1)-cosine)[:,None]


class _MLPPath(torch.autograd.Function):
    """One gradient path over shared, already-computed estimator activations."""
    @staticmethod
    def forward(ctx, x, w1, w2, saved):
        ctx.dtypes = (x.dtype,w1.dtype,w2.dtype)
        ctx.save_for_backward(*saved[:-1])
        # Distinct storage also keeps the paths separate under AOT Autograd.
        return saved[-1].clone()

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient):
        x,w1,w2,pre,hidden = ctx.saved_tensors
        with torch.autocast(x.device.type, enabled=False):
            gradient = gradient.to(hidden.dtype)
            dz = torch.ops.aten.silu_backward(gradient @ w2,pre)
            dw2 = gradient.flatten(0,-2).T @ hidden.flatten(0,-2)
            dw1 = dz.flatten(0,-2).T @ x.flatten(0,-2)
            dx = dz @ w1 if ctx.needs_input_grad[0] else None
        return (None if dx is None else dx.to(ctx.dtypes[0]),
                dw1.to(ctx.dtypes[1]),dw2.to(ctx.dtypes[2]),None)


class BlockMLP:
    """One forward, structurally separate LM and auxiliary gradient paths.

    The auxiliary node takes x.detach(): it cannot schedule backbone backward.
    Returning None for x from a shared two-output node was insufficient to prune
    that graph and exposed an undefined-gradient CUDA backward failure.
    Saved activations are shared, with no estimator forward recomputation.
    """
    @staticmethod
    def apply(x,w1,w2):
        dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
        with torch.no_grad(), torch.autocast(x.device.type,enabled=False):
            xc,w1c,w2c = (v.to(dtype) for v in (x,w1,w2))
            pre = F.linear(xc,w1c)
            hidden = F.silu(pre)
            out = F.linear(hidden,w2c)
        # A tuple is metadata to autograd; only the three explicit tensors define
        # gradient edges. _MLPPath saves its tensor contents via save_for_backward.
        saved = (xc,w1c,w2c,pre,hidden,out)
        return (_MLPPath.apply(x,w1,w2,saved),
                _MLPPath.apply(x.detach(),w1,w2,saved))


class AnticipatoryHead(nn.Module):
    def __init__(self, config, settings, *, stream=False, first=False):
        super().__init__()
        d,w = config.hidden_size,settings.width
        self.stream = stream
        self.w1 = nn.Linear(d,w,bias=False)
        self.w2 = nn.Linear(w,d,bias=False)
        self.delta = (nn.Sequential(nn.Linear(2*w,w,bias=False),nn.SiLU(),nn.Linear(w,w,bias=False))
                      if stream and not first else None)
        self.alpha = nn.Parameter(torch.full((1,d),float(settings.alpha_init)))
        for module in self.modules():
            if isinstance(module,nn.Linear):nn.init.normal_(module.weight,std=config.initializer_range)

    def stream_step(self, u, previous):
        projected = self.w1(u.detach())
        state = projected if self.delta is None else previous+self.delta(torch.cat((projected,previous),dim=-1))
        return self.w2(state),state

    def estimates(self,u,plan=None):
        if self.stream:raise ValueError('P5 requires its previous depth-stream state')
        return (self.w2(F.silu(self.w1(u))),)

    def routed(self,u,settings):
        # Downstream classification has no auxiliary loss. Opt into full task
        # gradients only through its wrapper; pretraining isolation is unchanged.
        if getattr(self, 'task_finetuning', False):
            value = self.estimates(u)[0]
            return value, value
        if settings.isolate_estimator:
            value = self.estimates(u.detach())[0]
            return value.detach(), self.estimates(u.detach())[0] if settings.aux_recompute else value
        if settings.aux_recompute:
            return self.estimates(u)[0],self.estimates(u.detach())[0]
        return BlockMLP.apply(u,self.w1.weight,self.w2.weight)
