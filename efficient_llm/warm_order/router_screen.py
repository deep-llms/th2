"""Router screen: routers trained jointly with the body (docs/ROUTER_SCREEN_20261123.md).

Same AttnRes sandwich body and data as the domain screen (E->T->M->T->H, stored blocks 0=T, 1..3=middles,
alternating single-domain DCLM/FineWeb-Edu batches). Arms:
  seq_switch2/3  per-sequence top-1 choice among 2/3 middles from a causal prefix router; MoE-style gradient
                 through a forward-preserving multiplier p/p.detach() on the chosen middle's outputs
  seq_cf2        per-sequence choice among 2 middles; router supervised online by counterfactual losses
                 (first CF_ROWS sequences of each batch scored under every route, no grad)
  order_cf       E->T->Ma->T->Mb->T->H vs E->T->Mb->T->Ma->T->H chosen per sequence, counterfactual router
  both_mix       same 5-application body with a uniformly random order per sequence (control)
  tok_moe2/3     token-level top-1 MoE over the middle MLP (2/3 experts = blocks 1..3 MLPs, shared attention)
Training samples routes from the router distribution (exploration); evaluation takes the argmax.
Balance: Switch-style K * sum_k usage_k * P_k with usage an EMA over steps (spans both domains, so
domain-consistent routing is not penalized).
"""
import torch
from torch import nn
from torch.nn import functional as F

from efficient_llm.dot_pool.model import MLP
from .domain_screen import DomainScreen
from .embedding_anchor import EmbeddingAnchor

ARMS = {
    'seq_switch2': dict(kind='seq', k=2, learn='switch'),
    'seq_switch3': dict(kind='seq', k=3, learn='switch'),
    'seq_cf2': dict(kind='seq', k=2, learn='cf'),
    'order_cf': dict(kind='order', k=2, learn='cf'),
    'both_mix': dict(kind='order', k=2, learn='random'),
    'tok_moe2': dict(kind='token', k=2, learn='switch'),
    'tok_moe3': dict(kind='token', k=3, learn='switch'),
    # standard MoE controls (docs/MOE_DYN_ROUND_20261123.md): deterministic top-k in training and evaluation,
    # gate multiplies the expert output (Switch / Mixtral), per-batch Switch balance loss; fresh experts
    'moe2_top1': dict(kind='moe', k=2, top=1, ff_div=1, learn='switch'),
    'moe4_top1': dict(kind='moe', k=4, top=1, ff_div=1, learn='switch'),
    'moe8_top2': dict(kind='moe', k=8, top=2, ff_div=2, learn='switch'),
}
SEQ_ROUTES = ((0, 1, 0), (0, 2, 0), (0, 3, 0))
ORDER_ROUTES = ((0, 1, 0, 2, 0), (0, 2, 0, 1, 0))
ROUTED = 'routed'
BALANCE = 0.01
USAGE_MOMENTUM = 0.95
CF_ROWS = 8


class RouterScreen(DomainScreen):
    def __init__(self, config, arm):
        EmbeddingAnchor.__init__(self, config)  # identical base init to the domain screen arms
        if arm not in ARMS:
            raise ValueError('Unknown router arm')
        self.arm, self.spec = arm, ARMS[arm]
        k = self.spec['k']
        self.attnres_query = nn.Parameter(torch.zeros(2 * len(self.blocks) + 1, config.d))
        self.router = nn.Parameter(torch.zeros(k, config.d))  # zero init: uniform routing at start
        self.register_buffer('usage', torch.full((k,), 1. / k))
        if self.spec['kind'] == 'moe':
            self.experts = nn.ModuleList(MLP(config.d, config.ff // self.spec['ff_div']) for _ in range(k))
            for module in self.experts.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=.02)
        self.generator = None
        self.aux = None
        self.scale = None  # per-sequence multiplier on middle outputs, set only inside train_loss
        self.last_choice = None

    # ----- routes -------------------------------------------------------------------------------
    def routes(self):
        return SEQ_ROUTES[:self.spec['k']] if self.spec['kind'] == 'seq' else ORDER_ROUTES

    def forced_paths(self):
        if self.spec['kind'] == 'moe':
            return {}
        if self.spec['kind'] == 'token':
            return {f'expert{k}': ('expert', k) for k in range(self.spec['k'])}
        return {f'route{k}': r for k, r in enumerate(self.routes())}

    def seq_logits(self, x):
        feature = self.embed(x[:, :self.c.prefix]).float().mean(1).detach()
        return F.rms_norm(feature, (self.c.d,), eps=1e-6) @ self.router.float().t()

    def sample(self, probs):
        if self.generator is None:
            raise RuntimeError('Training routes need an explicit generator')
        u = torch.rand(probs.shape[:-1], generator=self.generator, device=probs.device)
        return (probs.cumsum(-1) < u.unsqueeze(-1)).sum(-1).clamp(max=probs.shape[-1] - 1)

    def choose(self, x):
        """Per-sequence choice: sampled in training (uniform for both_mix), argmax in evaluation."""
        k = self.spec['k']
        if self.spec['learn'] == 'random':
            if self.training:
                return self.sample(torch.full((x.shape[0], k), 1. / k, device=x.device))
            return torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        probs = torch.softmax(self.seq_logits(x), -1)
        return self.sample(probs) if self.training else probs.argmax(-1)

    # ----- body ---------------------------------------------------------------------------------
    def _body(self, x, route, scale=None, moe=False, force=None):
        pos = torch.arange(x.shape[1], device=x.device).expand_as(x)
        seg = torch.zeros_like(x)
        sources = [self.embed(x)]
        for identity in route:
            block = self.blocks[identity]
            a = block.self_attn(block.input_layernorm(self._mix(sources, 2 * identity)), pos, seg)
            sources.append(a * scale.to(a.dtype) if identity and scale is not None else a)
            u = block.post_attention_layernorm(self._mix(sources, 2 * identity + 1))
            if moe and identity == 1:
                m = self._moe(u, force)
            else:
                m = block.mlp(u)
                if identity and scale is not None:
                    m = m * scale.to(m.dtype)
            sources.append(m)
        return self._mix(sources, 2 * len(self.blocks))

    def _moe(self, u, force=None):
        if self.spec['kind'] == 'moe':
            return self._moe_std(u, force)
        k = self.spec['k']
        experts = [self.blocks[1 + i].mlp for i in range(k)]
        if force is not None:
            return experts[force](u)
        probs = torch.softmax(F.rms_norm(u.float(), (self.c.d,), eps=1e-6) @ self.router.float().t(), -1)
        choice = self.sample(probs.detach()) if self.training else probs.argmax(-1)
        onehot = F.one_hot(choice, k).to(u.dtype)
        gate = (probs / probs.detach()).to(u.dtype)  # forward exactly 1; MoE gradient to the router
        out = sum(onehot[..., i:i + 1] * gate[..., i:i + 1] * experts[i](u) for i in range(k))  # exact top-1
        if self.training:
            self._record(probs.reshape(-1, k), choice.reshape(-1))
        self.last_choice = choice.detach()
        return out

    def _moe_std(self, u, force=None):
        k, top = self.spec['k'], self.spec['top']
        if force is not None:
            return self.experts[force](u)
        probs = torch.softmax(F.rms_norm(u.float(), (self.c.d,), eps=1e-6) @ self.router.float().t(), -1)
        value, index = probs.topk(top, -1)
        if top > 1:
            value = value / value.sum(-1, keepdim=True)  # Mixtral: renormalized over the chosen experts
        weights = sum(F.one_hot(index[..., j], k).to(probs.dtype) * value[..., j:j + 1] for j in range(top))
        out = sum(weights[..., i:i + 1].to(u.dtype) * self.experts[i](u) for i in range(k))  # exact top-k
        if self.training:
            assigned = sum(F.one_hot(index[..., j], k).float() for j in range(top)).reshape(-1, k)
            frac = assigned.mean(0) / top
            with torch.no_grad():
                self.usage.mul_(USAGE_MOMENTUM).add_((1 - USAGE_MOMENTUM) * frac)
            balance = BALANCE * k * (frac.detach() * probs.reshape(-1, k).mean(0)).sum()  # per-batch Switch loss
            self.aux = balance if self.aux is None else self.aux + balance
        self.last_choice = index[..., 0].detach()
        return out

    def _record(self, probs, choice):
        k = self.spec['k']
        frac = torch.bincount(choice, minlength=k).float() / choice.numel()
        with torch.no_grad():
            self.usage.mul_(USAGE_MOMENTUM).add_((1 - USAGE_MOMENTUM) * frac)
        balance = BALANCE * k * (self.usage.detach() * probs.mean(0)).sum()
        self.aux = balance if self.aux is None else self.aux + balance

    def hidden(self, x, path, *, alpha=1., donor=None, skip=None):
        from .embedding_anchor_runtime import anchor_device
        if alpha != 1. or donor is not None or skip is not None:
            raise ValueError('Router screen takes no anchor, donor or skip options')
        if x.device.type != anchor_device() or self.embed.weight.device.type != anchor_device():
            raise RuntimeError('Explicit matching anchor device scope required')
        if self.spec['kind'] in ('token', 'moe'):
            if path == ROUTED:
                return self._body(x, SEQ_ROUTES[0], moe=True)
            if isinstance(path, tuple) and len(path) == 2 and path[0] == 'expert' and 0 <= path[1] < self.spec['k']:
                return self._body(x, SEQ_ROUTES[0], moe=True, force=path[1])
            raise ValueError('Unregistered token-arm path')
        if tuple(path) not in self.routes():
            raise ValueError('Unregistered route (routed sequence arms go through token_losses)')
        return self._body(x, tuple(path), scale=self.scale)

    # ----- losses -------------------------------------------------------------------------------
    def token_losses(self, x, path, **kwargs):
        if self.spec['kind'] in ('token', 'moe') or path != ROUTED:
            return super().token_losses(x, path, **kwargs)
        if self.training and torch.is_grad_enabled():
            raise RuntimeError('Use train_loss for routed sequence training')
        choice = self.choose(x)
        out = None
        for k, route in enumerate(self.routes()):
            idx = (choice == k).nonzero(as_tuple=True)[0]
            if idx.numel():
                part = super().token_losses(x[idx], route, **kwargs)
                out = part.new_zeros((x.shape[0],) + part.shape[1:]) if out is None else out
                out[idx] = part
        self.last_choice = choice
        return out

    def train_loss(self, x):
        """Mean LM loss over the batch plus auxiliary router losses; returns (lm, aux or None, choice or None)."""
        self.aux = None
        spec, k = self.spec, self.spec['k']
        if spec['kind'] in ('token', 'moe'):
            lm = self.nll(x, ROUTED).mean()
            return lm, self.aux, None
        aux = None
        if spec['learn'] in ('switch', 'cf'):
            logits = self.seq_logits(x)
            probs = torch.softmax(logits, -1)
            choice = self.sample(probs.detach())
            self._record(probs, choice)
            aux = self.aux
            if spec['learn'] == 'cf':
                with torch.no_grad():
                    cf = torch.stack([super(RouterScreen, self).token_losses(x[:CF_ROWS], r).mean(1)
                                      for r in self.routes()], 1)
                aux = aux + F.cross_entropy(logits[:CF_ROWS], cf.argmin(1))
        else:
            choice = self.choose(x)
        losses = []
        for i, route in enumerate(self.routes()):
            idx = (choice == i).nonzero(as_tuple=True)[0]
            if not idx.numel():
                continue
            xi = x[idx]  # token ids: no gradient flows through this gather
            if spec['learn'] == 'switch':
                p = torch.softmax(self.seq_logits(xi), -1)[:, i]
                self.scale = (p / p.detach())[:, None, None]  # forward exactly 1; MoE gradient to the router
            try:
                losses.append(super().token_losses(xi, route).mean(1))
            finally:
                self.scale = None
        lm = torch.cat(losses).mean()
        return lm, aux, choice
