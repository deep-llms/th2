"""DOT sandwich with a token-level router, trained from scratch without warm-up (docs/DOT_SANDWICH_20261123.md).

User design (docs/EMBEDDING_ANCHOR_AND_TIED_TRANSLATOR_20261120.md, sections 3 and 8): tied translator T at every
boundary, a library of middles M_1..M_K, and a router placed BEFORE T that reads the current state and picks the
next middle j; T receives the recipient identity c_j (fixed sin-cos code per module, plus OUT for the readout).
Here every token decides its own path at every step:

    e -> T(e, c_j1) -> M_j1 -> T(., c_j2) -> M_j2 -> ... -> M_jD -> T(., c_OUT) -> H

Stored blocks: 0 = T, 1..3 = middles (identical base init to dense4). Modes: route (router per step, top-1 argmax,
Switch gate p on the chosen middle, balance loss), fixed (M1, M2, M3, ... cyclic), random (content-independent).
Residual: plain stream, or multi (each sublayer reads a softmax mix of the token's earlier outputs).
Middle attention is computed for all tokens per middle (keys/values of all positions are needed) and the chosen
output is selected; middle MLPs are dispatched sparsely with a deterministic permutation.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .dot_token import permute
from .embedding_anchor import EmbeddingAnchor

T_ID, MIDDLES = 0, (1, 2, 3)  # default library; spec 'middles' may extend it with fresh blocks
BALANCE = .01
RECIPIENT_MAX = .5  # per-channel bound of the id relative to T's normalized input (unit RMS)
RL_EPSILON, RL_ENTROPY = .05, .01  # rl: uniform exploration share and entropy bonus
CF_ROWS = 4  # cf: sequences per batch whose tokens are re-run with every middle at every step
PRIOR_BETA = 3.  # prior: logit bonus for the fixed order's middle at each step (start at the fixed sandwich)
CF_TAU = .05  # cf soft labels: softmax(-(loss - min loss) / CF_TAU) over middles
RL_WEIGHT, RL_CLIP = .1, 3.  # rl2: policy-gradient weight and advantage clip
# Multi-path router training (user, 24 Nov): judge whole paths by the FINAL loss and use several paths per token.
MPG_PATHS, MPG_ROWS, MPG_WEIGHT = 4, 8, .1  # mpg: sampled full paths per token, sequences per batch, PG weight
BRANCH_ROWS = 4  # branch: sequences per batch re-run with every middle at every step, greedy continuation


def recipient_codes(n, d):
    """Fixed sin-cos identity vectors (like absolute position codes) for module ids 0..n-1, unit RMS."""
    pos = torch.arange(n, dtype=torch.float32)[:, None]
    freq = torch.exp(-math.log(10000.) * torch.arange(0, d, 2, dtype=torch.float32) / d)
    code = torch.zeros(n, d)
    code[:, 0::2], code[:, 1::2] = torch.sin(pos * freq), torch.cos(pos * freq)
    return code


class DotSandwich(EmbeddingAnchor):
    def __init__(self, config, arm, spec):
        super().__init__(config)  # embed, blocks 0..3, head: identical base init to dense4
        self.arm, self.spec, self.sw = arm, spec, spec['sandwich']
        n_mid = self.sw.get('middles', 3)
        if n_mid > 3:  # extra middles: fresh blocks created after the shared base init
            from efficient_llm.dot_pool.model import Block, Config as BlockConfig
            bc = BlockConfig(d=config.d, heads=config.heads, kv_heads=config.kv_heads, head_dim=config.head_dim, ff=config.ff)
            for _ in range(n_mid - 3):
                block = Block(bc, True)
                for module in block.modules():
                    if isinstance(module, nn.Linear):
                        nn.init.normal_(module.weight, std=.02)
                self.blocks.append(block)
        self.middles = tuple(range(1, 1 + n_mid))
        d, depth, k = config.d, self.sw['depth'], len(self.middles)
        if self.sw['residual'] == 'multi':
            if n_mid != 3:
                raise ValueError('multi residual is implemented for the 3-middle library')
            self.attnres_query = nn.Parameter(torch.zeros(2 * 4 + 1, d))  # (block, sublayer) + readout
            self.attnres_router = nn.Parameter(torch.zeros(depth, d))
        # small random init: no exact argmax ties (CPU/GPU tie-breaking could differ)
        self.dot_router = nn.Parameter(torch.randn(depth, k, d) * .02)
        codes = recipient_codes(k + 1, d)
        self.register_buffer('codes', codes / codes.pow(2).mean(-1, keepdim=True).sqrt(), persistent=False)  # unit RMS
        # Recipient id = fixed sin-cos code (unit RMS) scaled by a learned per-channel gain (zero init), added AFTER
        # T's input norms. History: a d x d projection of the raw code (a01: std .02 init; a02: zero init) let Adam
        # grow the id to many times the token signal within a few hundred steps and broke every id arm (NLL 5.2-7.6).
        # The id is code * RECIPIENT_MAX * tanh(gain): never larger than RECIPIENT_MAX per channel vs the unit-RMS input.
        self.recipient = nn.Parameter(torch.zeros(d))
        self.aux = None
        self.trace = None
        self.generator = None  # rl: explicit sampling generator during training
        self.forced = None  # cf: {step: (B, L) choice} overrides while collecting counterfactual losses
        self.rollout = None  # mpg: list collecting (step, detached state, sampled choice) of a no-grad rollout
        self.test_depth = None  # evaluation-time depth override (extra steps reuse the last trained router)
        self._logp, self._logits, self._choices = [], [], []

    # ----- helpers -------------------------------------------------------------------------------
    @staticmethod
    def _key(v):
        return F.rms_norm(v.float(), (v.shape[-1],), eps=1e-6)

    @staticmethod
    def _mix(values, keys, query):
        weights = torch.softmax((keys * query.float()).sum(-1), dim=0).to(values.dtype)
        return (weights.unsqueeze(-1) * values).sum(0)

    def _router_logits(self, state, step):
        step_r = min(step, self.dot_router.shape[0] - 1)
        logits = F.rms_norm(state.float(), (self.c.d,), eps=1e-6) @ self.dot_router[step_r].float().t()
        if self.sw.get('prior'):  # start at the fixed sandwich: bonus for M(step mod 3) (one of the first 3 middles)
            logits = logits + PRIOR_BETA * F.one_hot(torch.tensor(step % 3, device=logits.device), logits.shape[-1]).float()
        return logits

    def _choose(self, state, step, shape):
        k, mode = len(self.middles), self.sw['mode']
        if self.rollout is not None:  # mpg rollout: sample from the policy with uniform exploration, record it
            probs = torch.softmax(self._router_logits(state, step), -1)
            explore = (1 - RL_EPSILON) * probs + RL_EPSILON / k
            u = torch.rand(shape, generator=self.generator, device=state.device)
            choice = (explore.cumsum(-1) < u.unsqueeze(-1)).sum(-1).clamp(max=k - 1)
            self.rollout.append((step, state.detach(), choice))
            return choice, probs.gather(-1, choice.unsqueeze(-1))
        if self.forced is not None and step in self.forced:
            choice = self.forced[step]
            if mode in ('route', 'cf', 'branch', 'mpg'):
                probs = torch.softmax(self._router_logits(state, step), -1)
                return choice, probs.gather(-1, choice.unsqueeze(-1))
            return choice, None
        if mode == 'rl':
            logp = torch.log_softmax(self._router_logits(state, step), -1)
            if self.training and torch.is_grad_enabled():
                if self.generator is None:
                    raise RuntimeError('rl routing needs an explicit generator')
                explore = (1 - RL_EPSILON) * logp.exp() + RL_EPSILON / k  # exploit the policy, explore new paths
                u = torch.rand(shape, generator=self.generator, device=state.device)
                choice = (explore.detach().cumsum(-1) < u.unsqueeze(-1)).sum(-1).clamp(max=k - 1)
                self._logp.append(logp.gather(-1, choice.unsqueeze(-1)).squeeze(-1))
                entropy = -(logp.exp() * logp).sum(-1).mean()
                self.aux = -RL_ENTROPY * entropy if self.aux is None else self.aux - RL_ENTROPY * entropy
            else:
                choice = logp.argmax(-1)
            return choice, None
        if mode == 'fixed':
            return torch.full(shape, step % k, dtype=torch.long, device=state.device), None
        if mode == 'random':
            g = torch.Generator(device='cpu').manual_seed(7919 * (step + 1))
            return torch.randint(0, k, shape, generator=g).to(state.device), None
        logits = self._router_logits(state, step)
        probs = torch.softmax(logits, -1)
        choice = probs.argmax(-1)
        if mode in ('cf', 'branch') and self.training and torch.is_grad_enabled():
            self._logits.append(logits)
            self._choices.append(choice.detach())
        if self.training and torch.is_grad_enabled():
            frac = F.one_hot(choice, k).float().reshape(-1, k).mean(0)
            balance = self.sw.get('balance', BALANCE) * k * (frac.detach() * probs.reshape(-1, k).mean(0)).sum()
            self.aux = balance if self.aux is None else self.aux + balance
        return choice, probs.gather(-1, choice.unsqueeze(-1))

    def _condition(self, code_index, dtype):
        """Recipient identity per token (added to T's normalized sublayer inputs); code_index (B, L) in 0..3 (3 = OUT)."""
        if not self.sw['recipient']:
            return None
        return (self.codes[code_index] * (RECIPIENT_MAX * torch.tanh(self.recipient))).to(dtype)

    def _middle_mlp(self, u, choice):
        shape = u.shape
        flat, idx = u.reshape(-1, shape[-1]), choice.reshape(-1)
        perm = torch.argsort(idx, stable=True)
        inverse = torch.argsort(perm, stable=True)
        counts = torch.bincount(idx, minlength=len(self.middles)).tolist()
        parts = torch.split(permute(flat, perm, inverse), counts)
        outs = [self.blocks[self.middles[j]].mlp(self.blocks[self.middles[j]].post_attention_layernorm(p)) if len(p) else p
                for j, p in enumerate(parts)]
        return permute(torch.cat(outs), inverse, perm).reshape(shape)

    # ----- forward -------------------------------------------------------------------------------
    def hidden(self, x, path, *, alpha=1., donor=None, skip=None):
        from .embedding_anchor_runtime import anchor_device
        if alpha != 1. or donor is not None or skip is not None:
            raise ValueError('DOT sandwich takes no anchor, donor or skip options')
        if x.device.type != anchor_device() or self.embed.weight.device.type != anchor_device():
            raise RuntimeError('Explicit matching anchor device scope required')
        if tuple(path) != (0, 1, 2, 3):
            raise ValueError('DOT sandwich takes the native library path only')
        if self.forced is None:
            self._logp, self._logits, self._choices = [], [], []
        return (self._multi if self.sw['residual'] == 'multi' else self._plain)(x)

    def extra_loss(self, x, token_losses):
        # Counterfactual / rollout forwards reset self.aux; keep the main forward's balance / entropy terms.
        # (Before 24 Nov 07:00 SGT the cf/mpg/branch arms of a03-a05 lost their main-path balance loss here.)
        saved = self.aux
        try:
            return self._extra_loss(x, token_losses)
        finally:
            self.aux = saved

    def _extra_loss(self, x, token_losses):
        """Router learning signals that need the per-token LM losses of the forward just run.

        rl: REINFORCE on every step's sampled choice with the token's own loss as cost (batch-normalized advantage).
        cf: re-run the first CF_ROWS sequences with every middle forced at one step (other steps keep the
            current choices), label each token with the middle giving the lowest loss, cross-entropy on the router.
        """
        mode, p0 = self.sw['mode'], self.c.prefix - 1
        if mode == 'rl' and self._logp:
            logp = torch.stack(self._logp).sum(0)[:, p0:-1]  # positions that predict a scored token
            if self.sw.get('v2'):  # per-position baseline over the batch, clipped, down-weighted
                adv = (token_losses - token_losses.mean(0, keepdim=True)) / (token_losses.std(0, keepdim=True) + 1e-6)
                return RL_WEIGHT * (logp * adv.clamp(-RL_CLIP, RL_CLIP).detach()).mean()
            adv = (token_losses - token_losses.mean()) / (token_losses.std() + 1e-6)
            return (logp * adv.detach()).mean()
        if mode == 'mpg' and self.training:
            # Several sampled full paths per token, each judged by the token's final loss; leave-one-out baseline
            # (RLOO): a path is reinforced when it ends better than the other paths of the same token.
            if self.generator is None:
                raise RuntimeError('mpg rollouts need an explicit generator')
            xs = x[:min(MPG_ROWS, x.shape[0])]
            records, losses = [], []
            with torch.no_grad():
                for _ in range(MPG_PATHS):
                    self.rollout = []
                    try:
                        losses.append(super().token_losses(xs, (0, 1, 2, 3)))
                    finally:
                        records.append(self.rollout)
                        self.rollout = None
            L = torch.stack(losses)  # (paths, rows, L - prefix)
            adv = L - (L.sum(0, keepdim=True) - L) / (MPG_PATHS - 1)
            adv = (adv / (adv.std(0, keepdim=True) + 1e-6)).clamp(-RL_CLIP, RL_CLIP)
            total = 0
            for g, record in enumerate(records):
                for step, state, choice in record:
                    logp = torch.log_softmax(self._router_logits(state, step), -1).gather(-1, choice.unsqueeze(-1)).squeeze(-1)
                    total = total + (logp[:, p0:-1] * adv[g]).mean()
            return MPG_WEIGHT * total / MPG_PATHS
        if mode == 'branch' and self._logits:
            # Branch at every step over every middle, let the router continue greedily from the new state, and
            # use the final losses as soft targets for that step (search-improved policy targets, cost depth x K).
            rows = min(BRANCH_ROWS, x.shape[0])
            logits = [g[:rows] for g in self._logits]
            current = [c[:rows] for c in self._choices]
            targets = []
            with torch.no_grad():
                for step in range(self.sw['depth']):
                    alts = []
                    for j in range(len(self.middles)):
                        self.forced = {s: current[s] for s in range(step)}
                        self.forced[step] = torch.full_like(current[step], j)
                        try:
                            alts.append(super().token_losses(x[:rows], (0, 1, 2, 3)))
                        finally:
                            self.forced = None
                    q = torch.stack(alts, -1)  # (rows, L - prefix, middles): final loss after branching
                    targets.append(torch.softmax(-(q - q.min(-1, keepdim=True).values) / CF_TAU, -1))
            k = len(self.middles)
            return sum(F.cross_entropy(logits[s][:, p0:-1].reshape(-1, k), targets[s].reshape(-1, k))
                       for s in range(self.sw['depth'])) / self.sw['depth']
        if mode == 'cf' and self._logits:
            rows = min(CF_ROWS, x.shape[0])
            logits = [g[:rows] for g in self._logits]
            current = [c[:rows] for c in self._choices]
            labels = []
            with torch.no_grad():  # counterfactual forwards only; the router loss below keeps its graph
                for step in range(self.sw['depth']):
                    alts = []
                    for j in range(len(self.middles)):
                        self.forced = {s: (torch.full_like(current[s], j) if s == step else current[s])
                                       for s in range(self.sw['depth'])}
                        try:
                            alts.append(super().token_losses(x[:rows], (0, 1, 2, 3)))
                        finally:
                            self.forced = None
                    stacked = torch.stack(alts, -1)  # (rows, L - prefix, middles)
                    if self.sw.get('soft'):  # soft targets: prefer lower loss without hard winner-take-all
                        labels.append(torch.softmax(-(stacked - stacked.min(-1, keepdim=True).values) / CF_TAU, -1))
                    else:
                        labels.append(stacked.argmin(-1))
            k = len(self.middles)
            total = sum(F.cross_entropy(logits[step][:, p0:-1].reshape(-1, k),
                                        labels[step].reshape(-1, k) if self.sw.get('soft') else labels[step].reshape(-1))
                        for step in range(self.sw['depth']))
            return total / self.sw['depth']
        return None

    def _plain(self, x):
        pos = torch.arange(x.shape[1], device=x.device).expand_as(x)
        seg = torch.zeros_like(x)
        self.aux = None
        T = self.blocks[T_ID]
        h = self.embed(x)
        for step in range(self.test_depth or self.sw['depth']):
            choice, gate = self._choose(h, step, x.shape)  # router BEFORE T reads the current state
            if self.trace is not None:
                self.trace.append(choice.detach())
            c = self._condition(choice, h.dtype)
            h = self._t(T, h, c, pos, seg)
            onehot = F.one_hot(choice, len(self.middles)).to(h.dtype)
            a = 0
            for j, b in enumerate(self.middles):
                a = a + onehot[..., j:j + 1] * self.blocks[b].self_attn(self.blocks[b].input_layernorm(h), pos, seg)
            h = h + (a * gate.to(a.dtype) if gate is not None else a)
            m = self._middle_mlp(h, choice)
            h = h + (m * gate.to(m.dtype) if gate is not None else m)
        out = torch.full(x.shape, len(self.middles), dtype=torch.long, device=x.device)
        return self._t(T, h, self._condition(out, h.dtype), pos, seg)

    @staticmethod
    def _t(T, h, c, pos, seg):
        """Tied translator conditioned on the recipient: c is added to both normalized sublayer inputs, not the stream."""
        u = T.input_layernorm(h)
        h = h + T.self_attn(u if c is None else u + c, pos, seg)
        u = T.post_attention_layernorm(h)
        return h + T.mlp(u if c is None else u + c)

    def _multi(self, x):
        pos = torch.arange(x.shape[1], device=x.device).expand_as(x)
        seg = torch.zeros_like(x)
        self.aux = None
        T, q = self.blocks[T_ID], self.attnres_query
        e = self.embed(x)
        sources, keys = [e], [self._key(e)]

        def mix(query):
            return self._mix(torch.stack(sources), torch.stack(keys), query)

        def add(v):
            sources.append(v)
            keys.append(self._key(v))

        def translator(c):
            u = T.input_layernorm(mix(q[0]))
            add(T.self_attn(u if c is None else u + c, pos, seg))
            u = T.post_attention_layernorm(mix(q[1]))
            add(T.mlp(u if c is None else u + c))

        for step in range(self.sw['depth']):
            choice, gate = self._choose(mix(self.attnres_router[step]), step, x.shape)
            if self.trace is not None:
                self.trace.append(choice.detach())
            translator(self._condition(choice, e.dtype))
            onehot = F.one_hot(choice, len(self.middles)).to(e.dtype)
            a = 0
            for j, b in enumerate(self.middles):
                a = a + onehot[..., j:j + 1] * self.blocks[b].self_attn(self.blocks[b].input_layernorm(mix(q[2 * b])), pos, seg)
            add(a * gate.to(a.dtype) if gate is not None else a)
            query = (onehot.unsqueeze(-1) * q[3:8:2].to(e.dtype)).sum(-2)  # chosen middle's MLP query per token
            m = self._middle_mlp(mix(query), choice)
            add(m * gate.to(m.dtype) if gate is not None else m)
        out = torch.full(x.shape, len(self.middles), dtype=torch.long, device=x.device)
        translator(self._condition(out, e.dtype))
        return mix(q[8])
