"""DOT sandwich with a joint path + STOP policy per token (docs/DOT_STOP_SANDWICH_20261124.md).

User (24 Nov): order can change without losing quality, so STOP may find cheaper exits when paths are diverse; path and
STOP learning should inform each other; option: decide a token's depth early. At every step each still-active token
picks one of the K middles OR STOP (one policy, K+1 actions); STOP freezes the token (copy-through; its state still
serves as keys/values) and the final tied translator T(., c_OUT) + head read every token out once at the end.
Variants: fixed_path (only continue/STOP is learned on the fixed middle order), budget (a step-0 head picks the number
of middles 1..D, then the path is routed within it), stop disabled (pure routing at depth D).

Training: the body learns on the policy's SAMPLED path (every depth gets exposure); the policy learns from G sampled
full paths per token judged by the final loss plus lambda x middles used, with a leave-one-out baseline (RLOO).
Anti-collapse: SAC-style adaptive entropy coefficient towards a target entropy, and an exploration schedule
(epsilon-uniform mixing decaying over the first 30% of training). Evaluation is greedy.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .dot_sandwich import PRIOR_BETA, RECIPIENT_MAX, DotSandwich, T_ID, recipient_codes

ROLL_PATHS, ROLL_ROWS, PG_WEIGHT, ADV_CLIP = 4, 8, .3, 3.
EPS0, EPS_MIN, EPS_DECAY = .2, .02, .3
ENT_ALPHA0, ENT_LR, ENT_TARGET_FRAC = .01, .01, .5
LAM_RAMP = .25


class _MLPOnly(nn.Module):
    """T without attention: keeps block 0's MLP and its norm (same init values)."""

    def __init__(self, block):
        super().__init__()
        self.post_attention_layernorm, self.mlp = block.post_attention_layernorm, block.mlp


class DotSandwichStop(DotSandwich):
    def __init__(self, config, arm, spec):
        super().__init__(config, arm, spec)
        sw, d, k = self.sw, config.d, len(self.middles)
        self.stop_index = k
        self.stop_router = nn.Parameter(torch.randn(sw['depth'], k + 1, d) * .02)
        if sw.get('budget'):
            self.budget_router = nn.Parameter(torch.randn(sw['depth'], d) * .02)  # budget b = index + 1 middles
        self.register_buffer('log_alpha', torch.tensor(math.log(ENT_ALPHA0)))
        # f01 (25 Nov): give the shared translator T per-recipient capacity (docs/DOT_FIXED_IMPROVE_20261125.md)
        n_rec = k + 1  # recipients: the K middles and OUT
        if sw.get('t_ff_div'):  # light T: a narrower FFN, so compute moves to the middles
            from efficient_llm.dot_pool.model import MLP
            self.blocks[T_ID].mlp = MLP(d, config.ff // sw['t_ff_div'])
            for module in self.blocks[T_ID].mlp.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=.02)
        if (sw.get('t_width') or sw.get('t_heads')) and (sw.get('t_ff_div') or sw.get('film') or sw.get('lora_rank')):
            raise ValueError('compact T (t_width / t_heads) does not combine with t_ff_div, film or lora_rank')
        if sw.get('t_mlp'):  # f03: T is only a (recipient-conditioned) nonlinear MLP; block 0's attention is dropped
            if any(sw.get(key) for key in ('t_width', 't_heads', 't_ff_div', 'lora_rank')):
                raise ValueError('t_mlp combines only with film')
            self.blocks[T_ID] = _MLPOnly(self.blocks[T_ID])
            if sw.get('t_ff'):  # f09: a narrower MLP-only T (fresh init), so T costs a fraction of a block
                from efficient_llm.dot_pool.model import MLP
                self.blocks[T_ID].mlp = MLP(d, sw['t_ff'])
                for module in self.blocks[T_ID].mlp.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.normal_(module.weight, std=.02)
            if sw.get('t_unshared'):  # f14 ablation: one separate MLP-only T per position (not shared), same compute
                import copy
                from efficient_llm.dot_pool.model import MLP, RMSNorm
                if not (sw.get('fixed_path') and not sw.get('stop', True)):
                    raise ValueError('t_unshared is defined for the static fixed path')
                self.t_list = nn.ModuleList()
                for _ in range(sw['depth'] + 1):
                    t = copy.deepcopy(self.blocks[T_ID])
                    for module in t.mlp.modules():
                        if isinstance(module, nn.Linear):
                            nn.init.normal_(module.weight, std=.02)
                    self.t_list.append(t)
                self.blocks[T_ID] = self.t_list[0]  # no orphan shared T: block 0 is position 0's own T
        if sw.get('t_width') or sw.get('t_heads'):  # f02: a compact T (narrower width through a bottleneck, fewer heads)
            from efficient_llm.dot_pool.model import Block, Config as BlockConfig
            d_t = sw.get('t_width', d)
            heads_t = sw.get('t_heads', d_t // config.head_dim)
            self.blocks[T_ID] = Block(BlockConfig(d=d_t, heads=heads_t, kv_heads=sw.get('t_kv', max(1, heads_t // 3)),
                                                  head_dim=config.head_dim, ff=sw.get('t_ff', config.ff * d_t // d)), True)
            for module in self.blocks[T_ID].modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=.02)
            if d_t != d:  # T works in a d_t bottleneck: h + up(T(down h) - down h)
                self.t_down, self.t_up = nn.Linear(d, d_t, bias=False), nn.Linear(d_t, d, bias=False)
                for module in (self.t_down, self.t_up):
                    nn.init.normal_(module.weight, std=.02)
                codes = recipient_codes(k + 1, d_t)
                self.register_buffer('t_codes', codes / codes.pow(2).mean(-1, keepdim=True).sqrt(), persistent=False)
                self.t_recipient = nn.Parameter(torch.zeros(d_t))
        if sw.get('film'):  # per-recipient scale and shift of T's two normalised sublayer inputs, zero at init
            # (sublayer, scale|shift, recipient, d); an MLP-only T has one sublayer
            self.t_film = nn.Parameter(torch.zeros(1 if sw.get('t_mlp') else 2, 2, n_rec, d))
        if sw.get('lora_rank'):  # per-recipient low-rank residual on T's MLP, B zero at init
            r = sw['lora_rank']
            self.t_lora_a = nn.Parameter(torch.randn(n_rec, r, d) / math.sqrt(d))
            self.t_lora_b = nn.Parameter(torch.zeros(n_rec, d, r))
        if spec.get('dot_t_norm') and not spec.get('huginn_block'):  # L9: DOT-native, only T re-standardizes the state
            from efficient_llm.dot_pool.model import RMSNorm  # (blocks stay pre-norm; T is the interface between modules)
            if not (self._static_path() and sw.get('t_mlp')):
                raise ValueError('dot_t_norm is defined for the static fixed path with an MLP-only T')
            for t in (getattr(self, 't_list', None) or [self.blocks[T_ID]]):
                t.norm_out = RMSNorm(d)
                if spec.get('t_norm_only'):  # L18 ablation: T keeps only its output norm (no MLP), i.e. a plain fixed
                    t.mlp = None  # loop of the middles with a per-position RMSNorm between modules
                    t.post_attention_layernorm = None
        elif spec.get('t_norm_only'):
            raise ValueError('t_norm_only is an ablation of dot_t_norm')
        if spec.get('huginn_block'):  # f23: Huginn's sandwich norms on the middles' and T's residuals (+ sqrt(d) embedding)
            from efficient_llm.dot_pool.model import RMSNorm, enable_sandwich
            if not (self._static_path() and sw.get('t_mlp')):
                raise ValueError('huginn_block is defined for the static fixed path with an MLP-only T')
            for j in self.middles:
                enable_sandwich(self.blocks[j], d)
            for t in (getattr(self, 't_list', None) or [self.blocks[T_ID]]):
                t.norm_out = RMSNorm(d)
        self.fast_fixed = True  # static fixed path: run only the chosen middle (False: the generic all-middle path)
        self.eval_order = None  # static path: an explicit middle index per step (overrides the fixed / random order)
        self.train_progress = 0.
        self.records = None  # rollout collector: list of (logits_fn args) for the policy gradient
        self.answer_mask = None  # synthetic tasks: bool (L - prefix,) of scored answer positions in the suffix
        self.last_used = None

    # ----- translator --------------------------------------------------------------------------
    def _tc(self, T, h, idx, pos, seg):
        """T conditioned on the recipient: idx is a (B, L) index tensor, or a plain int when every token has the same
        recipient (static fixed path: the id / FiLM / LoRA terms broadcast; same values as the per-token form).
        ID code, optional FiLM and per-recipient LoRA."""
        static = isinstance(idx, int)

        def film(sub, u):
            if static:
                return self.t_film[sub, 0, idx].to(u.dtype), self.t_film[sub, 1, idx].to(u.dtype)
            onehot = F.one_hot(idx, self.stop_index + 1).to(u.dtype)  # one-hot matmul, not a gather (slow backward)
            gb = onehot @ self.t_film[sub].transpose(0, 1).reshape(self.t_film.shape[2], -1).to(u.dtype)
            return gb.unflatten(-1, (2, -1)).unbind(-2)

        if self.spec.get('t_norm_only'):  # L18 ablation: T is only its output norm
            return T.norm_out(h)
        if self.sw.get('t_mlp'):  # MLP-only T: h + MLP(norm(h) + id), optionally FiLM per recipient
            c = self._condition(idx, h.dtype)
            u = T.post_attention_layernorm(h)
            u = u if c is None else u + c
            if self.sw.get('film'):
                g, b = film(0, u)
                u = u * (1 + g) + b
            out = h + T.mlp(u)
            return T.norm_out(out) if 'norm_out' in T._modules else out  # f23 sandwich norm on T's residual
        if hasattr(self, 't_down'):  # compact T in a bottleneck
            z = self.t_down(h)
            c = (self.t_codes[idx] * (RECIPIENT_MAX * torch.tanh(self.t_recipient))).to(z.dtype) if self.sw['recipient'] else None
            return h + self.t_up(self._t(T, z, c, pos, seg) - z)
        if not (self.sw.get('film') or self.sw.get('lora_rank')):
            return self._t(T, h, self._condition(idx, h.dtype), pos, seg)
        c = self._condition(idx, h.dtype)

        def prep(u, sub):
            if c is not None:
                u = u + c
            if self.sw.get('film'):
                g, b = film(sub, u)
                u = u * (1 + g) + b
            return u

        h = h + T.self_attn(prep(T.input_layernorm(h), 0), pos, seg)
        u = prep(T.post_attention_layernorm(h), 1)
        out = T.mlp(u)
        if self.sw.get('lora_rank'):
            onehot = None if static else F.one_hot(idx, self.t_lora_a.shape[0]).to(u.dtype)
            for r in ([idx] if static else range(self.t_lora_a.shape[0])):
                z = (u @ self.t_lora_a[r].to(u.dtype).t()) @ self.t_lora_b[r].to(u.dtype).t()
                out = out + (z if static else onehot[..., r:r + 1] * z)
        return h + out

    def _static_path(self):
        """Fixed path without STOP or budget: one legal middle per step, known before the forward."""
        return bool(self.sw.get('fixed_path')) and not self.sw.get('stop', True) and not self.sw.get('budget')

    # ----- policy ------------------------------------------------------------------------------
    def _allowed(self, step, shape, device):
        k, sw = self.stop_index, self.sw
        allowed = torch.ones(*shape, k + 1, dtype=torch.bool, device=device)
        if step == 0 or not sw.get('stop', True) or sw.get('budget'):
            allowed[..., k] = False  # at least one middle; STOP decided by the budget head or disabled
        if sw.get('fixed_path'):
            keep = torch.zeros(k + 1, dtype=torch.bool, device=device)
            keep[self.sw.get('fixed_middle', step % k)] = True  # fixed_middle: always the same middle (a pure loop)
            keep[k] = True
            allowed &= keep
        return allowed

    def _policy_logits(self, state, step, allowed):
        step_r = min(step, self.stop_router.shape[0] - 1)  # extra evaluation steps reuse the last trained step
        logits = F.rms_norm(state.float(), (self.c.d,), eps=1e-6) @ self.stop_router[step_r].float().t()
        if self.sw.get('prior'):  # fixed-order prior: training starts as the fixed sandwich, the router may deviate
            logits = logits + PRIOR_BETA * F.one_hot(torch.tensor(step % self.stop_index, device=logits.device),
                                                     logits.shape[-1])
        return logits.masked_fill(~allowed, float('-inf'))

    def _lam(self):
        """Compute penalty; with sw['lam_start'] it is 0 before that training fraction and ramps up over LAM_RAMP."""
        lam, start = float(self.sw.get('lam', 0.)), self.sw.get('lam_start')
        if start is None:
            return lam
        return lam * min(1., max(0., (self.train_progress - start) / LAM_RAMP))

    def _epsilon(self):
        return max(EPS_MIN, EPS0 * (1 - self.train_progress / EPS_DECAY))

    def _sample(self, logits, allowed, eps):
        probs = torch.softmax(logits, -1)
        uniform = allowed.float() / allowed.float().sum(-1, keepdim=True)
        mix = (1 - eps) * probs + eps * uniform
        u = torch.rand(mix.shape[:-1], generator=self.generator, device=mix.device)
        return (mix.cumsum(-1) < u.unsqueeze(-1)).sum(-1).clamp(max=mix.shape[-1] - 1)

    # ----- forward -----------------------------------------------------------------------------
    def hidden(self, x, path, *, alpha=1., donor=None, skip=None):
        from .embedding_anchor_runtime import anchor_device
        if alpha != 1. or donor is not None or skip is not None:
            raise ValueError('DOT stop sandwich takes no anchor, donor or skip options')
        if x.device.type != anchor_device() or self.embed.weight.device.type != anchor_device():
            raise RuntimeError('Explicit matching anchor device scope required')
        if tuple(path) != (0, 1, 2, 3):
            raise ValueError('DOT stop sandwich takes the native library path only')
        if self.sw['residual'] != 'plain':
            raise ValueError('stop sandwich uses the plain residual stream')
        sample = self.training and (torch.is_grad_enabled() or self.records is not None)
        if self.sw.get('body_greedy') and self.records is None:
            sample = False  # the body trains on the greedy path; only the policy rollouts sample
        if sample and self.generator is None:
            raise RuntimeError('sampled paths need an explicit generator')
        return self._run(x, sample)

    def _run(self, x, sample):
        pos = torch.arange(x.shape[1], device=x.device).expand_as(x)
        seg = None  # one sequence per row (no packed documents in this input)
        self.aux = None
        T, k = self.blocks[T_ID], self.stop_index
        depth = self.test_depth or self.sw['depth']
        eps = self._epsilon() if sample else 0.
        h = self.embed(x)
        if self.spec.get('huginn_block') or self.spec.get('emb_sqrt_d'):  # sqrt(d) embedding scale (Vaswani et al. 2017)
            h = h * math.sqrt(self.c.d)
        active = torch.ones(x.shape, dtype=torch.bool, device=x.device)
        used = torch.zeros(x.shape, dtype=torch.long, device=x.device)
        entropies = []
        budget = None
        if self.sw.get('budget'):
            blog = F.rms_norm(h.float(), (self.c.d,), eps=1e-6) @ self.budget_router.float().t()
            ball = torch.ones(*x.shape, self.budget_router.shape[0], dtype=torch.bool, device=x.device)
            bidx = self._sample(blog, ball, eps) if sample else blog.argmax(-1)
            budget = bidx + 1
            if self.records is not None:
                self.records.append(('budget', h.detach(), bidx, active.clone(), ball))
            if sample and torch.is_grad_enabled():
                entropies.append(torch.distributions.Categorical(logits=blog).entropy())
        if self._static_path() and self.fast_fixed:
            # one statically known middle per step and no STOP: no policy, no masks; only that block runs, and the
            # recipient terms broadcast (same values as the generic path below; tests/test_dot_fixed_improve.py)
            t_list = getattr(self, 't_list', None)
            # shuffle only in a gradient-carrying training forward: no-grad passes (CPU/GPU parity checks, scoring) use
            # the canonical order, so they never depend on a device-specific random stream
            shuffle = (self.sw.get('random_order') and self.training and torch.is_grad_enabled()
                       and self.eval_order is None)
            for step in range(depth):
                if self.eval_order is not None:  # an explicit middle order (evaluation of order robustness)
                    j = self.eval_order[step]
                elif shuffle:  # f18: one random middle per step for the whole batch (order must be learned anew)
                    j = int(torch.randint(k, (), generator=self.generator, device=self.generator.device))
                else:
                    j = self.sw.get('fixed_middle', step % k)
                if self.trace is not None:
                    self.trace.append(torch.full(x.shape, j, dtype=torch.long, device=x.device))
                if t_list is not None:  # unshared ablation: position step's own T (deeper sweeps reuse the last)
                    T = t_list[min(step, len(t_list) - 2)]
                h_t = self._tc(T, h, j, pos, seg)
                block = self.blocks[self.middles[j]]
                h = block.ffn(block.attention(h_t, pos, seg))  # = h_m + mlp(norm(h_m)), h_m = h_t + attn(norm(h_t))
            self.last_used = torch.full(x.shape, depth, dtype=torch.long, device=x.device)
            return self._tc(t_list[-1] if t_list is not None else T, h, k, pos, seg)
        if hasattr(self, 't_list'):
            raise ValueError('the unshared-T ablation runs on the static fixed path only')
        for step in range(depth):
            allowed = self._allowed(step, x.shape, x.device)
            logits = self._policy_logits(h, step, allowed)
            a = self._sample(logits, allowed, eps) if sample else logits.argmax(-1)
            if budget is not None:
                a = torch.where(step >= budget, torch.full_like(a, k), a)
            decide = active if budget is None else active & (step < budget)
            if self.records is not None:
                self.records.append((step, h.detach(), a, decide.clone(), allowed))
            if sample and torch.is_grad_enabled():
                lp = torch.log_softmax(logits, -1)
                ent = -(lp.exp() * lp.masked_fill(~allowed, 0.)).sum(-1)
                entropies.append(ent[decide])
            go = active & (a != k)
            if self.trace is not None:
                self.trace.append(torch.where(go, a, torch.full_like(a, k)).detach())
            choice = torch.where(go, a, torch.zeros_like(a))
            h_t = self._tc(T, h, choice, pos, seg)
            onehot = F.one_hot(choice, k).to(h.dtype)
            att = 0
            for j, b in enumerate(self.middles):
                att = att + onehot[..., j:j + 1] * self.blocks[b].self_attn(self.blocks[b].input_layernorm(h_t), pos, seg)
            h_m = h_t + att
            h_m = h_m + self._middle_mlp(h_m, choice)
            h = torch.where(go.unsqueeze(-1), h_m, h)  # stopped tokens are frozen (copy-through)
            used = used + go.long()
            active = go
        if entropies and torch.is_grad_enabled():
            flat = torch.cat([e.reshape(-1) for e in entropies])
            if flat.numel():
                entropy = flat.mean()
                n_allowed = float(k + 1)
                target = ENT_TARGET_FRAC * math.log(n_allowed)
                with torch.no_grad():
                    self.log_alpha.add_(ENT_LR * (target - float(entropy))).clamp_(math.log(1e-4), math.log(1.))
                self.aux = -self.log_alpha.exp().detach() * entropy
        self.last_used = used.detach()
        out = torch.full(x.shape, k, dtype=torch.long, device=x.device)
        return self._tc(T, h, out, pos, seg)

    # ----- policy learning -----------------------------------------------------------------------
    def _extra_loss(self, x, token_losses):
        """RLOO over ROLL_PATHS sampled full paths per token: cost = final loss + lambda x middles used.

        With sw['context_credit'] (synthetic tasks, where only the answer is scored) unscored context positions are
        also policy decisions; each gets its row's mean answer loss as outcome (plus its own compute penalty)."""
        if not self.training:
            return None
        if self.sw.get('fixed_path') and not self.sw.get('stop', True):
            return None  # a single legal action at every step: no policy to learn, skip the rollouts
        lam, p0 = self._lam(), self.c.prefix - 1
        lo = 0 if self.sw.get('context_credit') else p0
        xs = x[:min(self.sw.get('roll_rows', ROLL_ROWS), x.shape[0])]
        records, costs = [], []
        with torch.no_grad():
            for _ in range(ROLL_PATHS):
                self.records = []
                try:
                    tl = super(DotSandwich, self).token_losses(xs, (0, 1, 2, 3))  # EmbeddingAnchor.token_losses
                    if self.answer_mask is not None:  # unscored queries inside the suffix: row-mean answer outcome
                        m = self.answer_mask.to(tl.device).float()
                        tl = torch.where(m.bool(), tl, ((tl * m).sum(1, keepdim=True) / m.sum()).expand_as(tl))
                    if lo < p0:
                        tl = torch.cat([tl.mean(1, keepdim=True).expand(-1, p0 - lo), tl], 1)
                    costs.append(tl + lam * self.last_used[:, lo:-1].float())
                finally:
                    records.append(self.records)
                    self.records = None
        C = torch.stack(costs)  # (paths, rows, L - 1 - lo)
        adv = C - (C.sum(0, keepdim=True) - C) / (ROLL_PATHS - 1)
        adv = (adv / (adv.std(0, keepdim=True) + 1e-6)).clamp(-ADV_CLIP, ADV_CLIP)
        total = 0
        for g, record in enumerate(records):
            for step, state, action, decide, allowed in record:
                if step == 'budget':
                    logits = F.rms_norm(state.float(), (self.c.d,), eps=1e-6) @ self.budget_router.float().t()
                else:
                    logits = self._policy_logits(state, step, allowed)
                logp = torch.log_softmax(logits, -1).gather(-1, action.unsqueeze(-1)).squeeze(-1)
                mask = decide[:, lo:-1].float()
                total = total + (logp[:, lo:-1].masked_fill(mask == 0, 0.) * adv[g] * mask).sum() / mask.sum().clamp(min=1.)
        return PG_WEIGHT * total / ROLL_PATHS
