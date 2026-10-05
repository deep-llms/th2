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


class BlockMLP(torch.autograd.Function):
    """One forward, two VJPs: LM reaches input+weights; auxiliary reaches weights.

    Cast as autocast Linear does, retain its forward activations, and use ATen's
    SiLU backward. Backward needs no estimator forward or hidden-state hooks.
    Higher-order derivatives are not used by this training path.
    """
    @staticmethod
    def forward(ctx, x, w1, w2):
        ctx.dtypes = (x.dtype, w1.dtype, w2.dtype)
        dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
        with torch.autocast(x.device.type, enabled=False):
            x, w1, w2 = (v.to(dtype) for v in (x,w1,w2))
            pre = F.linear(x,w1)
            hidden = F.silu(pre)
            out = F.linear(hidden,w2)
        ctx.save_for_backward(x,w1,w2,pre,hidden)
        ctx.set_materialize_grads(False)
        # Independent storage preserves the two VJPs under AOT Autograd too.
        # An alias/view can be deduplicated by compilation, merging the routes.
        return out, out.clone()

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, lm, aux):
        x,w1,w2,pre,hidden = ctx.saved_tensors
        with torch.autocast(x.device.type, enabled=False):
            total = aux if lm is None else lm if aux is None else lm+aux
            if total is None:return None,None,None
            total = total.to(hidden.dtype)
            dz = torch.ops.aten.silu_backward(total @ w2,pre)
            dw2 = total.flatten(0,-2).T @ hidden.flatten(0,-2)
            dw1 = dz.flatten(0,-2).T @ x.flatten(0,-2)
            dx = None if lm is None else torch.ops.aten.silu_backward(lm.to(hidden.dtype) @ w2,pre) @ w1
        return (None if dx is None else dx.to(ctx.dtypes[0]),dw1.to(ctx.dtypes[1]),dw2.to(ctx.dtypes[2]))


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
        if settings.isolate_estimator:
            value = self.estimates(u.detach())[0]
            return value.detach(), self.estimates(u.detach())[0] if settings.aux_recompute else value
        if settings.aux_recompute:
            return self.estimates(u)[0],self.estimates(u.detach())[0]
        return BlockMLP.apply(u,self.w1.weight,self.w2.weight)
