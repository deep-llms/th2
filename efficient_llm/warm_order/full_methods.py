"""Full published looped methods, ported exactly at our scale (docs/FULL_LOOPED_METHODS_20261126.md).

Huginn (Geiping et al. 2025; seal-rg/recurrent-pretraining 1ea7220e, HF tomg-group-umd/huginn-0125),
Ouro / LoopLM (arXiv 2510.25741 v6; HF ByteDance/Ouro-1.4B modeling_ouro.py 574fa66c) and
Mixture-of-Recursions (Bae et al. 2025; raymin0223/mixture_of_recursions 82c97f29).

Each class keeps its method's block, loop, input injection, head / tying, init, loss, sampler and
inference rule. Only sizes are ours (d, heads, kv heads, FFN, vocab, context). Every model accepts
optional `segments` (B, T) document ids for packed rows: attention never crosses documents and rotary
positions restart at each document, as in the rest of the code base. segments=None means one
sequence per row.

The interface matches the loop-screen trainer: hidden / readout_hidden / token_losses / nll, plus
train_loss(x, path, step, total) (the method's own objective) and output_weight() (unembedding).
"""
from contextlib import nullcontext
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from efficient_llm.dot_pool.model import attend, causal_mask, document_positions

EVAL_STATE_SEED = 20261126  # Huginn s0 at evaluation: a fixed CPU generator per forward (reproducible, same law)


# ----------------------------------------------------------------------------------------------- shared
class RMSNorm(nn.Module):
    """Learnable-weight RMSNorm computed in FP32 (Huginn norms.py / HF Llama / Ouro)."""
    def __init__(self, d, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            y = x.float()
            y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        return y.to(x.dtype) * self.weight


def positions(x, segments):
    if segments is None:
        return torch.arange(x.shape[1], device=x.device).expand_as(x)
    return document_positions(segments)


def rope_halves(z, pos, theta):
    """Llama rotate_half RoPE (Ouro, MoR): z (B, T, H, D), pos (B, T)."""
    d = z.shape[-1]
    inv = 1. / (theta ** (torch.arange(0, d, 2, device=z.device, dtype=torch.int64).float() / d))
    ang = pos.float()[..., None] * inv
    ang = torch.cat([ang, ang], -1)[:, :, None, :]
    cos, sin = ang.cos().to(z.dtype), ang.sin().to(z.dtype)
    return z * cos + torch.cat([-z[..., d // 2:], z[..., :d // 2]], -1) * sin


def rope_pairs(z, pos, theta):
    """Huginn interleaved-pair RoPE (apply_rotary_emb_complex_like), computed in FP32: z (B, T, H, D)."""
    d = z.shape[-1]
    inv = 1. / (theta ** (torch.arange(0, d, 2, dtype=torch.float32, device=z.device) / d))
    ang = pos.float()[..., None] * inv  # (B, T, D/2)
    cos, sin = ang.cos()[:, :, None, :], ang.sin()[:, :, None, :]
    with torch.autocast(z.device.type, enabled=False):
        r = z.float().unflatten(-1, (-1, 2))
        out = torch.stack([r[..., 0] * cos - r[..., 1] * sin, r[..., 1] * cos + r[..., 0] * sin], -1).flatten(3)
    return out.type_as(z)


def suffix_ce(model, h, x, weight):
    """Per-token CE on the scored suffix (positions >= prefix), chunked and checkpointed in training as the
    EmbeddingAnchor readout does. h: final hidden (B, T, d) aligned with x."""
    p = model.c.prefix
    h, y = h[:, p - 1:-1], x[:, p:]
    hh, yy = h.reshape(-1, h.shape[-1]), y.reshape(-1)

    def ce(a, w, target):
        return F.cross_entropy(F.linear(a, w).float(), target, reduction='none')

    out = []
    for s in range(0, len(yy), 1024):
        args = hh[s:s + 1024], weight, yy[s:s + 1024]
        out.append(checkpoint(ce, *args, use_reentrant=False) if model.training and torch.is_grad_enabled()
                   else ce(*args))
    return torch.cat(out).reshape_as(y)


class _Base(nn.Module):
    def __init__(self, c, arm, spec):
        super().__init__()
        if c.d != c.heads * c.head_dim or c.heads % c.kv_heads or c.head_dim % 2:
            raise ValueError('Invalid dimensions')
        self.c, self.arm, self.spec, self.f = c, arm, spec, spec['full']
        self.backend = 'fused' if spec.get('fast_attn') else 'sdpa'
        self.passes = 3  # unused (loop-screen trainer sets it); the method's own loop count lives in self.f
        self.aux = None
        self.eval_mode = {}

    def output_weight(self):
        return self.embed.weight

    def token_losses(self, x, path=None, skip=None, segments=None):
        if x.shape[1] <= self.c.prefix:
            raise ValueError('No scored suffix')
        return suffix_ce(self, self.readout_hidden(self.hidden(x, path, segments=segments)), x, self.output_weight())

    def nll(self, x, path=None, **kw):
        return self.token_losses(x, path, **kw).mean(1)

    def train_loss(self, x, path, step, total):
        return self.token_losses(x, path).mean()


def _no_decay(name, p):
    return p.ndim < 2 or 'norm' in name or 'bias' in name


def adamw_groups(model, weight_decay):
    """HF-Trainer style AdamW groups (MoR's trainer; used for Ouro too): decay on weight matrices and the
    embedding, none on norms and biases."""
    decay = [p for n, p in model.named_parameters() if not _no_decay(n, p)]
    rest = [p for n, p in model.named_parameters() if _no_decay(n, p)]
    return [dict(params=decay, weight_decay=weight_decay), dict(params=rest, weight_decay=0.)]


# ----------------------------------------------------------------------------------------------- Huginn
class HuginnAttention(nn.Module):
    """GH model_dynamic.py CausalSelfAttention: packed Wqkv [Q | K | V], k/v repeated to every query head,
    then the learnable q/k bias, then interleaved RoPE (base 50000), causal SDPA, proj."""
    def __init__(self, c, backend):
        super().__init__()
        self.c, self.backend = c, backend
        self.Wqkv = nn.Linear(c.d, (c.heads + 2 * c.kv_heads) * c.head_dim, bias=False)
        self.qk_bias = nn.Parameter(torch.zeros(2, 1, c.heads, c.head_dim))
        self.proj = nn.Linear(c.d, c.d, bias=False)

    def forward(self, x, pos, segments):
        c, (b, t, _) = self.c, x.shape
        q, k, v = self.Wqkv(x).split([c.heads * c.head_dim, c.kv_heads * c.head_dim, c.kv_heads * c.head_dim], 2)
        rep = c.heads // c.kv_heads
        q = q.view(b, t, c.heads, c.head_dim)
        k = k.view(b, t, c.kv_heads, c.head_dim).repeat_interleave(rep, 2)
        v = v.view(b, t, c.kv_heads, c.head_dim).repeat_interleave(rep, 2)
        q_bias, k_bias = self.qk_bias.split(1, 0)
        q, k = (q + q_bias).to(q.dtype), (k + k_bias).to(q.dtype)
        q, k = rope_pairs(q, pos, 50000.), rope_pairs(k, pos, 50000.)
        return self.proj(attend(q, k, v, segments, self.backend))


class HuginnMLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fc = nn.Linear(c.d, 2 * c.ff, bias=False)  # [gate | up]
        self.proj = nn.Linear(c.ff, c.d, bias=False)

    def forward(self, x):
        a, b = self.fc(x).chunk(2, -1)
        return self.proj(F.silu(a) * b)


class SandwichBlock(nn.Module):
    """x = n2(x + Attn(n1 x)); x = n4(x + MLP(n3 x)) (Huginn). prenorm=True is the f22 ablation
    x = x + Attn(n1 x); x = x + MLP(n3 x) (no norms on the residual stream)."""
    def __init__(self, c, backend, prenorm=False):
        super().__init__()
        self.prenorm = prenorm
        self.norm_1, self.norm_3 = RMSNorm(c.d, 1e-6), RMSNorm(c.d, 1e-6)
        if not prenorm:
            self.norm_2, self.norm_4 = RMSNorm(c.d, 1e-6), RMSNorm(c.d, 1e-6)
        self.attn, self.mlp = HuginnAttention(c, backend), HuginnMLP(c)

    def forward(self, x, pos, segments):
        if self.prenorm:
            x = x + self.attn(self.norm_1(x), pos, segments)
            return x + self.mlp(self.norm_3(x))
        x = self.norm_2(self.attn(self.norm_1(x), pos, segments) + x)
        return self.norm_4(self.mlp(self.norm_3(x)) + x)


HUGINN_ABLATIONS = ('prenorm', 'adamw', 'zero_state', 'fixed_r', 'add_inject', 'no_emb_scale', 'full_bp')


class HuginnLM(_Base):
    """Recurrent-depth LM: e = prelude(sqrt(d) wte(x)); s0 ~ truncN(0, sqrt(2/5)) (per forward);
    r times s = core(adapter([s, e])); h = coda(ln_f(s)); logits = tied_head(ln_f(h)).

    f22 component analysis (docs/FULL_LOOPED_METHODS_20261126.md): f['ablate'] removes one component at a time
    (the default, no ablation, is the published method):
      prenorm      pre-norm blocks instead of the 4-norm sandwich
      adamw        AdamW (wd 0) with our warmup-stable-decay schedule instead of ELLISAdam + trapezoid
      zero_state   s0 = 0 instead of a random state per forward
      fixed_r      r = mean_recurrence every step with full backprop (no sampled depth, no truncation)
      add_inject   s = s + e (no adapter) instead of s = adapter([s, e])
      no_emb_scale no sqrt(d) scale on the embedding (and on s0)
      full_bp      L26 control: the same sampled depth sequence with gradients through every recurrence (no
                   truncation; each recurrence checkpointed so activations fit)"""
    def __init__(self, c, arm, spec):
        super().__init__(c, arm, spec)
        f = self.f
        self.ablate = tuple(f.get('ablate', ()))
        if set(self.ablate) - set(HUGINN_ABLATIONS):
            raise ValueError('Unknown Huginn ablation')
        self.mean_recurrence, self.backprop = f['mean_recurrence'], f['backprop']
        pre = 'prenorm' in self.ablate
        self.embed = nn.Embedding(c.vocab, c.d)
        self.prelude = nn.ModuleList(SandwichBlock(c, self.backend, pre) for _ in range(f['prelude']))
        if 'add_inject' not in self.ablate:
            self.adapter = nn.Linear(2 * c.d, c.d, bias=False)
        self.core = nn.ModuleList(SandwichBlock(c, self.backend, pre) for _ in range(f['core']))
        self.coda = nn.ModuleList(SandwichBlock(c, self.backend, pre) for _ in range(f['coda']))
        self.ln_f = RMSNorm(c.d, 1e-6)  # used twice
        self.emb_scale = 1. if 'no_emb_scale' in self.ablate else math.sqrt(c.d)
        self.std = math.sqrt(2 / (5 * c.d))
        self.generator = None  # set by build(): device generator for the training s0
        self.train_steps = None  # (n_no_grad, k_grad) for the current training step
        self.state_override = None  # tests / reference parity: an explicit s0
        self.reset_parameters()
        # L49 analysis (not part of the published method): DOT's readout translator T(s, e) = norm(s + W [n(s), n(e)])
        # in place of ln_f before the coda; built after reset_parameters so every Huginn tensor keeps its init
        self.coda_t = None
        if f.get('coda_t'):
            from .dot_recurrent import Translator
            self.coda_t = Translator(c.d, 0, True, True, linear=True)
            nn.init.normal_(self.coda_t.mlp.weight, std=.02)

    @torch.no_grad()
    def reset_parameters(self):
        """'takase' init (GH init.py): truncated N(+-3 std); std = sqrt(2/(5 d)) for Wqkv, adapter, embedding
        and both halves of the GLU fc; out projections std / sqrt(2 l), l = prelude + core r + coda."""
        f = self.f
        depth = f['prelude'] + f['core'] * self.mean_recurrence + f['coda']
        out = self.std / math.sqrt(2 * depth)

        def tn(w, std):
            nn.init.trunc_normal_(w, mean=0., std=std, a=-3 * std, b=3 * std)

        tn(self.embed.weight, self.std)
        if hasattr(self, 'adapter'):
            tn(self.adapter.weight, self.std)
        for block in [*self.prelude, *self.core, *self.coda]:
            tn(block.attn.Wqkv.weight, self.std)
            tn(block.mlp.fc.weight, self.std)
            tn(block.attn.proj.weight, out)
            tn(block.mlp.proj.weight, out)

    def sample_steps(self, step):
        """GH 'poisson-lognormal-filling', one draw per optimizer step (lockstep), CPU generator 514229 + step."""
        t, s = max(self.mean_recurrence - self.backprop, 0), self.backprop
        g = torch.Generator(device='cpu')
        g.manual_seed((514229 + step) % (2 ** 31 - 1))
        sigma = .5
        mu = math.log(t + s) - sigma ** 2 / 2
        rate = torch.zeros((1,)).log_normal_(mean=mu, std=sigma, generator=g)
        p = torch.poisson(torch.tensor([rate], dtype=torch.float), generator=g) + 1
        n = torch.clamp(p - s, min=0)
        k = torch.minimum(torch.as_tensor(s), p)
        return int(n), int(k)

    def initial_state(self, e):
        if self.state_override is not None:
            return self.state_override.to(e.device, e.dtype)
        if 'zero_state' in self.ablate:
            return torch.zeros_like(e)
        # Training draws from the device generator; every no-grad forward (evaluation, qualification parity in
        # either mode) from a fixed CPU generator, so CPU and device see the same s0 (f18 lesson).
        fresh = self.training and torch.is_grad_enabled() and self.generator is not None
        x = torch.empty(e.shape, dtype=torch.float32, device=self.generator.device if fresh else 'cpu')
        if fresh:
            g = self.generator
        else:
            g = torch.Generator(device='cpu')
            g.manual_seed(EVAL_STATE_SEED)
        nn.init.trunc_normal_(x, mean=0., std=self.std, a=-3 * self.std, b=3 * self.std, generator=g)
        return (x * self.emb_scale).to(e.device, e.dtype)

    def _core(self, s, e, pos, segments):
        s = s + e if 'add_inject' in self.ablate else self.adapter(torch.cat([s, e], -1))
        for block in self.core:
            s = block(s, pos, segments)
        return s

    def hidden(self, x, path=None, *, skip=None, segments=None):
        pos = positions(x, segments)
        e = self.embed(x) * self.emb_scale
        for block in self.prelude:
            e = block(e, pos, segments)
        s = self.initial_state(e)
        if self.training and torch.is_grad_enabled():
            n, k = self.train_steps if self.train_steps is not None else self.sample_steps(0)
        else:
            n, k = self.eval_mode.get('steps', self.mean_recurrence), 0
        dt = x.device.type
        # Autocast caches weight casts; a cast first made under no_grad has no grad_fn and would cut the core's
        # gradient in the later steps (GH paper A.2 caveat). Disable the cache for the no-grad steps.
        amp = (torch.autocast(dt, dtype=torch.get_autocast_dtype(dt), cache_enabled=False)
               if torch.is_autocast_enabled(dt) else nullcontext())
        with torch.no_grad(), amp:
            for _ in range(n):
                s = self._core(s, e, pos, segments)
        ckpt = 'full_bp' in self.ablate and self.training and torch.is_grad_enabled()
        for _ in range(k):
            s = checkpoint(self._core, s, e, pos, segments, use_reentrant=False) if ckpt else self._core(s, e, pos, segments)
        h = self.coda_t(s, e) if self.coda_t is not None else self.ln_f(s)
        for block in self.coda:
            h = block(h, pos, segments)
        return h

    def readout_hidden(self, h):
        return self.ln_f(h)

    def train_loss(self, x, path, step, total):
        self.train_steps = (0, self.mean_recurrence) if 'fixed_r' in self.ablate else self.sample_steps(step)
        if 'full_bp' in self.ablate:
            self.train_steps = (0, sum(self.train_steps))
        try:
            return self.token_losses(x, path).mean()
        finally:
            self.train_steps = None

    def eval_variants(self):
        r = sorted({1, 2, 3, 4, 6, 8, 16, 32, 2 * self.mean_recurrence} - {self.mean_recurrence})
        return [(f'r{v}', dict(steps=v)) for v in r]

    def optimizer(self, peak):
        if 'adamw' in self.ablate:  # the loop-screen baselines' optimizer
            return torch.optim.AdamW(self.parameters(), lr=peak, betas=(.9, .95), eps=1e-8, weight_decay=0)
        groups, wd = [], self.f.get('weight_decay', 4e-5)
        decay = [p for n, p in self.named_parameters() if not _no_decay(n, p)]
        rest = [p for n, p in self.named_parameters() if _no_decay(n, p)]
        groups = [dict(params=decay, weight_decay=wd), dict(params=rest, weight_decay=0.)]
        return ELLISAdam(groups, lr=peak, betas=(.9, .95))

    def lr(self, step, total, peak):
        """GH get_lr 'trapezoid': linear warmup, constant, linear cooldown to min_lr 0 (6.8% each, nebel yaml)."""
        if 'adamw' in self.ablate:  # our schedule: 128-step warmup, constant, linear decay to 0 over the last 20%
            from .loop_screen import lr_at
            return lr_at(step, total, peak)
        w = c = round(.068 * total)
        if step < w:
            return peak * step / w
        if step > total - c:
            return max(peak * (total - step) / c, 0.)
        return peak

    def applications(self):
        f = self.f
        return f['prelude'] + f['core'] * self.mean_recurrence + f['coda']


class ELLISAdam(torch.optim.Optimizer):
    """GH recpre/optim.py ELLISAdam with the nebel config: atan2 Adam (no eps), update clipping,
    running_init, bias correction, decoupled weight decay (p *= 1 - lr / init_lr * wd)."""
    def __init__(self, params, lr, betas=(.9, .95), eps=1e-6):
        super().__init__(params, dict(lr=lr, init_lr=lr, betas=betas, eps=eps, weight_decay=0.))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr, init_lr, wd, eps = group['lr'], group['init_lr'], group['weight_decay'], group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if not state:
                    state['step'] = 0
                    state['exp_avg'] = grad.clone()  # running_init
                    state['exp_avg_sq'] = grad.pow(2).clone()
                m, v = state['exp_avg'], state['exp_avg_sq']
                m.lerp_(grad, 1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                step_size = lr
                rms = grad.pow(2).div_(v.clamp_(min=eps ** 2)).mean().sqrt()  # update clipping (as optimi)
                step_size = step_size / rms.clamp(min=1.)
                state['step'] += 1
                bc1 = 1 - beta1 ** state['step']
                bc2_sqrt = math.sqrt(1 - beta2 ** state['step'])
                step_size = step_size / bc1
                p.mul_(1 - lr / init_lr * wd)
                p.add_(torch.atan2(m, v.sqrt().div_(bc2_sqrt)) * -step_size)


# ----------------------------------------------------------------------------------------------- Llama-style layers
class LlamaAttention(nn.Module):
    """q/k/v/o without bias, rotate_half RoPE, causal SDPA with GQA (Ouro, MoR)."""
    def __init__(self, c, backend, theta):
        super().__init__()
        self.c, self.backend, self.theta = c, backend, theta
        self.q_proj = nn.Linear(c.d, c.heads * c.head_dim, bias=False)
        self.k_proj = nn.Linear(c.d, c.kv_heads * c.head_dim, bias=False)
        self.v_proj = nn.Linear(c.d, c.kv_heads * c.head_dim, bias=False)
        self.o_proj = nn.Linear(c.heads * c.head_dim, c.d, bias=False)

    def qkv(self, x, pos):
        c, (b, t, _) = self.c, x.shape
        q = rope_halves(self.q_proj(x).view(b, t, c.heads, c.head_dim), pos, self.theta)
        k = rope_halves(self.k_proj(x).view(b, t, c.kv_heads, c.head_dim), pos, self.theta)
        return q, k, self.v_proj(x).view(b, t, c.kv_heads, c.head_dim)

    def forward(self, x, pos, segments, allowed=None):
        q, k, v = self.qkv(x, pos)
        if allowed is None:
            return self.o_proj(attend(q, k, v, segments, self.backend))
        # explicit (B, T, T) key mask (MoR causal routing: only tokens active at this recursion are keys)
        b, t = x.shape[:2]
        rep = self.c.heads // self.c.kv_heads
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.repeat_interleave(rep, 2).transpose(1, 2),
                                             v.repeat_interleave(rep, 2).transpose(1, 2), attn_mask=allowed[:, None])
        return self.o_proj(out.transpose(1, 2).reshape(b, t, -1))


class LlamaMLP(nn.Module):
    def __init__(self, d, f):
        super().__init__()
        self.gate_proj = nn.Linear(d, f, bias=False)
        self.up_proj = nn.Linear(d, f, bias=False)
        self.down_proj = nn.Linear(f, d, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class OuroLayer(nn.Module):
    """h += N2(Attn(N1 h)); h += N4(MLP(N3 h)) (modeling_ouro.py OuroDecoderLayer)."""
    def __init__(self, c, backend):
        super().__init__()
        self.self_attn = LlamaAttention(c, backend, 10000.)
        self.mlp = LlamaMLP(c.d, c.ff)
        self.input_layernorm, self.input_layernorm_2 = RMSNorm(c.d, 1e-6), RMSNorm(c.d, 1e-6)
        self.post_attention_layernorm, self.post_attention_layernorm_2 = RMSNorm(c.d, 1e-6), RMSNorm(c.d, 1e-6)

    def forward(self, h, pos, segments):
        h = h + self.input_layernorm_2(self.self_attn(self.input_layernorm(h), pos, segments))
        return h + self.post_attention_layernorm_2(self.mlp(self.post_attention_layernorm(h)))


class LlamaLayer(nn.Module):
    """Pre-norm Llama decoder layer (MoR's base model), RMSNorm eps 1e-5."""
    def __init__(self, c, backend):
        super().__init__()
        self.self_attn = LlamaAttention(c, backend, 10000.)
        self.mlp = LlamaMLP(c.d, c.ff)
        self.input_layernorm, self.post_attention_layernorm = RMSNorm(c.d, 1e-5), RMSNorm(c.d, 1e-5)

    def forward(self, h, pos, segments, allowed=None):
        h = h + self.self_attn(self.input_layernorm(h), pos, segments, allowed)
        return h + self.mlp(self.post_attention_layernorm(h))


def _init_normal(module, std=.02):
    """HF _init_weights: Linear / Embedding N(0, std), biases 0, norms 1."""
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=std)
            if getattr(m, 'bias', None) is not None:
                nn.init.zeros_(m.bias)


def wsd(step, total, peak, warmup, decay, floor, shape):
    """Warmup-stable-decay: linear warmup, constant, then `shape` decay to floor*peak over the last `decay`."""
    if step < warmup:
        return peak * (step + 1) / warmup
    start = int(total * (1 - decay))
    if step < start:
        return peak
    frac = (step - start) / max(1, total - start)
    if shape == 'cosine':
        return peak * (floor + (1 - floor) * .5 * (1 + math.cos(math.pi * frac)))
    return peak * (floor + (1 - floor) * (1 - frac))


# ----------------------------------------------------------------------------------------------- Ouro
class OuroLM(_Base):
    """Looped LM: the whole stack runs T times; after each step h = norm(h) feeds the (untied) head, the exit
    gate and the next step. Exit pdf p_t = lambda_t prod_{j<t}(1 - lambda_j), the last step takes the rest."""
    def __init__(self, c, arm, spec):
        super().__init__(c, arm, spec)
        self.steps = self.f['steps']
        self.embed = nn.Embedding(c.vocab, c.d)
        self.layers = nn.ModuleList(OuroLayer(c, self.backend) for _ in range(self.f['layers']))
        self.norm = RMSNorm(c.d, 1e-6)
        self.early_exit_gate = nn.Linear(c.d, 1)
        self.lm_head = nn.Linear(c.d, c.vocab, bias=False)
        _init_normal(self)

    def output_weight(self):
        return self.lm_head.weight

    def states(self, x, steps, segments=None):
        pos = positions(x, segments)
        h, hs, gates = self.embed(x), [], []
        for _ in range(steps):
            for layer in self.layers:
                h = layer(h, pos, segments)
            h = self.norm(h)
            hs.append(h)
            gates.append(self.early_exit_gate(h).squeeze(-1))
        return hs, gates

    @staticmethod
    def exit_pdf(gates):
        out, remaining = [], torch.ones_like(gates[0], dtype=torch.float32)
        for i, g in enumerate(gates):
            lam = torch.sigmoid(g.float())
            if i < len(gates) - 1:
                out.append(lam * remaining)
                remaining = remaining * (1 - lam)
            else:
                out.append(remaining)
        return torch.stack(out, -1)

    def hidden(self, x, path=None, *, skip=None, segments=None):
        hs, _ = self.states(x, self.eval_mode.get('steps', self.steps), segments)
        return hs[self.eval_mode.get('read', len(hs)) - 1]

    def readout_hidden(self, h):
        return h  # already normed every step

    def token_losses(self, x, path=None, skip=None, segments=None):
        mode = self.eval_mode
        if 'q' not in mode and not mode.get('mix'):
            return super().token_losses(x, path, segments=segments)
        hs, gates = self.states(x, mode.get('steps', self.steps), segments)
        p = self.c.prefix
        pdf = self.exit_pdf([g[:, p - 1:-1] for g in gates])  # (B, S, T)
        y = x[:, p:]
        if mode.get('mix'):  # HF labels path: CE of the pdf-weighted mixture of logits
            logits = sum(F.linear(h[:, p - 1:-1], self.lm_head.weight).float() * pdf[..., i:i + 1]
                         for i, h in enumerate(hs))
            return F.cross_entropy(logits.transpose(1, 2), y, reduction='none')
        # Q-exit: first step whose exit CDF reaches q (else the last)
        cdf = pdf.cumsum(-1)
        reach = cdf >= mode['q']
        first = torch.where(reach.any(-1), reach.long().argmax(-1), torch.full_like(y, len(hs) - 1))
        ce = torch.stack([suffix_ce(self, h, x, self.lm_head.weight) for h in hs], -1)
        self.last_exit = first
        return ce.gather(-1, first[..., None]).squeeze(-1)

    def beta(self, step, total):
        return .1 if step < .4 * total else .05  # Stage 1a (3T of 7.4T tokens) 0.1, then 0.05

    def train_loss(self, x, path, step, total):
        """Eq. 4 (v6): mean over scored tokens of sum_t p_t CE_t - beta H(p); nothing detached."""
        hs, gates = self.states(x, self.steps)
        p = self.c.prefix
        pdf = self.exit_pdf([g[:, p - 1:-1] for g in gates])
        ce = torch.stack([suffix_ce(self, h, x, self.lm_head.weight) for h in hs], -1)
        entropy = -(pdf * torch.log(pdf.clamp_min(1e-12))).sum(-1)
        return ((pdf * ce).sum(-1) - self.beta(step, total) * entropy).mean()

    def eval_variants(self):
        t = self.steps
        out = [(f'step{i}', dict(read=i)) for i in range(1, t)]
        out += [('mix', dict(mix=True))] + [(f'q{q}', dict(q=q)) for q in (.5, .7, .9)]
        out += [(f'steps{s}', dict(steps=s)) for s in (t + 1, 2 * t)]
        return out

    def optimizer(self, peak):
        return torch.optim.AdamW(adamw_groups(self, .1), lr=peak, betas=(.9, .95), eps=1e-8)

    def lr(self, step, total, peak):
        """WSD: our 128-step warmup (Ouro states none), constant, cosine to 0.1 peak over the last 19%."""
        return wsd(step, total, peak, 128, .19, .1, 'cosine')

    def applications(self):
        return self.f['layers'] * self.steps


# ----------------------------------------------------------------------------------------------- MoR
class MoRLM(_Base):
    """Middle-Cycle recursive LM with expert-choice routing (MoR):
    h = first(embed(x)); for r in recursions: router_r picks the top cap_r * T tokens among those picked at
    r - 1 (per sequence, kept in position order); they run the shared middle layers among themselves with
    their original positions and h_sel += g * Block(h_sel), g = alpha sigmoid(logit); h = last(h);
    logits = tied_head(norm(h)). Training adds coeff * sum_r BCE(logit, selected) over candidates."""
    def __init__(self, c, arm, spec):
        super().__init__(c, arm, spec)
        f = self.f
        self.recursions, self.alpha, self.coeff = f['recursions'], f.get('alpha', .1), f.get('coeff', 1e-3)
        self.capacity = f.get('capacity') or [(self.recursions - i) / self.recursions for i in range(self.recursions)]
        if any(a < b for a, b in zip(self.capacity, self.capacity[1:])):
            raise ValueError('capacity should be in descending order')
        self.embed = nn.Embedding(c.vocab, c.d)
        self.first = LlamaLayer(c, self.backend)
        self.middle = nn.ModuleList(LlamaLayer(c, self.backend) for _ in range(f['base_depth']))
        self.last = LlamaLayer(c, self.backend)
        self.norm = RMSNorm(c.d, 1e-5)
        self.routers = nn.ModuleList(nn.Linear(c.d, 1, bias=False) for _ in range(self.recursions))
        _init_normal(self)
        self.decisions = None  # last forward's per-recursion selections (qualify shares them CPU -> device)
        self.forced_decisions = None

    def _topk_step(self, r, h, prev, pos, segments):
        """One expert-choice recursion on the candidates `prev` (B, n) absolute indices, or all tokens."""
        b, t, d = h.shape
        k = max(1, int(self.capacity[r] * t))
        cand = h if prev is None else h.gather(1, prev[..., None].expand(-1, -1, d))
        logits = self.routers[r](cand)  # (B, n, 1), raw hidden / temp 1
        probs = torch.sigmoid(logits) * self.alpha
        if self.forced_decisions is not None:
            local = self.forced_decisions[r]
        else:
            local = torch.sort(torch.topk(probs, k, dim=1, sorted=False)[1], dim=1)[0]  # (B, k, 1)
        weights = probs.gather(1, local)
        aux = None
        if self.training:
            target = torch.zeros_like(logits).scatter_(1, local, 1.)
            aux = F.binary_cross_entropy_with_logits(logits.float().view(-1), target.float().view(-1), reduction='sum') / (b * logits.shape[1])
        sel = local if prev is None else prev.gather(1, local[..., 0])[..., None]  # absolute indices (B, k, 1)
        idx = sel.expand(-1, -1, d)
        z = h.gather(1, idx)
        zpos = pos.gather(1, sel[..., 0])
        zseg = None if segments is None else segments.gather(1, sel[..., 0])
        for layer in self.middle:
            z = layer(z, zpos, zseg)
        h = torch.scatter_add(h, 1, idx, z * weights)
        return h, sel[..., 0], local, aux

    def _causal_step(self, r, h, active, pos, segments):
        """Causal inference rule: a token runs recursion r iff it ran r - 1 and sigmoid(logit) >= 0.5; keys are
        the tokens active at r (recursion-wise KV), so no token reads the future."""
        b, t, _ = h.shape
        logits = self.routers[r](h)[..., 0]
        g = torch.sigmoid(logits) * self.alpha
        if self.forced_decisions is not None:
            now = self.forced_decisions[r]
        else:
            now = active & (torch.sigmoid(logits) >= .5)
        allowed = causal_mask(segments, t, h.device).expand(b, t, t) & now[:, None, :]
        allowed = allowed | torch.eye(t, dtype=torch.bool, device=h.device)  # inactive rows: discarded anyway
        z = h
        for layer in self.middle:
            z = layer(z, pos, segments, allowed)
        h = h + (now.to(h.dtype) * g.to(h.dtype))[..., None] * z
        return h, now

    def hidden(self, x, path=None, *, skip=None, segments=None):
        pos = positions(x, segments)
        h = self.first(self.embed(x), pos, segments)
        mode = 'topk' if self.training else self.eval_mode.get('route', 'causal')
        decisions, aux = [], []
        if mode == 'topk':
            prev = None
            for r in range(self.recursions):
                h, prev, local, a = self._topk_step(r, h, prev, pos, segments)
                decisions.append(local)
                if a is not None:
                    aux.append(a)
        else:
            active = torch.ones(x.shape, dtype=torch.bool, device=x.device)
            for r in range(self.recursions):
                h, active = self._causal_step(r, h, active, pos, segments)
                decisions.append(active)
        self.decisions = decisions
        self.aux_terms = aux
        return self.last(h, pos, segments)

    def readout_hidden(self, h):
        return self.norm(h)

    def train_loss(self, x, path, step, total):
        lm = self.token_losses(x, path).mean()
        return lm + self.coeff * sum(self.aux_terms)

    def eval_variants(self):
        return [('topk', dict(route='topk'))]

    def optimizer(self, peak):
        return torch.optim.AdamW(adamw_groups(self, .1), lr=peak, betas=(.9, .95), eps=1e-8)

    def lr(self, step, total, peak):
        """pretrain.py WSD: ~0.6% warmup (86 of 14161 MoR-3 steps), constant, linear decay to 0 over the last 20%."""
        return wsd(step, total, peak, max(1, round(86 / 14161 * total)), .2, 0., 'linear')

    def applications(self):
        return 2 + len(self.middle) * sum(self.capacity)


KINDS = dict(huginn=HuginnLM, ouro=OuroLM, mor=MoRLM)


def make_full_model(config, arm, spec):
    return KINDS[spec['full']['kind']](config, arm, spec)


@torch.no_grad()
def full_stats(model, rows, ids, device):
    """Method-specific measurement statistics (MoR causal applications; Ouro exit-step shares)."""
    model.eval()
    out = {}
    if isinstance(model, MoRLM):
        counts, n = torch.zeros(model.recursions, dtype=torch.float64), 0
        for i in range(0, len(ids), 4):
            x = rows.batch(ids[i:i + 4], device)
            model.hidden(x)
            for r, d in enumerate(model.decisions):
                counts[r] += float(d.double().sum())
            n += x.numel()
        share = (counts / n).tolist()
        out.update(causal_active_share=share, causal_mean_applications=2 + len(model.middle) * sum(share),
                   topk_applications=model.applications())
    if isinstance(model, OuroLM):
        for q in (.5, .7, .9):
            model.eval_mode = dict(q=q)
            steps = []
            for i in range(0, len(ids), 4):
                model.token_losses(rows.batch(ids[i:i + 4], device))
                steps.append(model.last_exit.double().flatten() + 1)
            model.eval_mode = {}
            s = torch.cat(steps)
            out[f'q{q}'] = dict(mean_steps=float(s.mean()),
                                share=[float((s == t + 1).double().mean()) for t in range(model.steps)])
    return out
