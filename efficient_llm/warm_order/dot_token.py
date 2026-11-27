"""Token-level DOT: at every step each token chooses the next block from a shared library (docs/DOT_TOKEN_20261123.md).

Library = the four stored blocks (identical base init to dense4). T steps per token. Modes:
  route   router per step reads the token state and picks one block (top-1 argmax, Switch gate p, balance loss)
  fixed   cyclic order s mod 4 for every token (same machinery, no routing): ablation for routing
  random  content-independent random block per token and step (fixed generator per forward): exploration control
Residual:
  multi   every sublayer reads a softmax mix over all earlier outputs of the token, with one position-free query
          per (block, sublayer), so a block reads the same way at any step (translator role)
  plain   standard residual stream
Attention of every library block is computed for all tokens (keys/values of all positions are needed) and the
output of the chosen block is selected; MLPs are dispatched sparsely with a deterministic permutation.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .embedding_anchor import EmbeddingAnchor

LIBRARY = 4
BALANCE = .01


class _Permute(torch.autograd.Function):
    """y = x[perm] along dim 0 with a gather-only backward (deterministic on CUDA)."""

    @staticmethod
    def forward(ctx, x, perm, inverse):
        ctx.save_for_backward(inverse)
        return x[perm]

    @staticmethod
    def backward(ctx, grad):
        (inverse,) = ctx.saved_tensors
        return grad[inverse], None, None


def permute(x, perm, inverse):
    return _Permute.apply(x, perm, inverse)


class DotToken(EmbeddingAnchor):
    def __init__(self, config, arm, spec):
        super().__init__(config)  # embed, blocks 0..3, head: identical base init to dense4
        self.arm, self.spec, self.dot = arm, spec, spec['dot']
        d, T = config.d, self.dot['steps']
        if self.dot['residual'] == 'multi':
            self.attnres_query = nn.Parameter(torch.zeros(2 * LIBRARY + 1, d))
            self.attnres_router = nn.Parameter(torch.zeros(T, d))  # router read query per step
        # small random init: no exact argmax ties (CPU/GPU tie-breaking could differ)
        self.dot_router = nn.Parameter(torch.randn(T, LIBRARY, d) * .02)
        self.aux = None
        self.trace = None  # set to [] to record the (B, L) block choice of every step

    # ----- helpers -------------------------------------------------------------------------------
    def _key(self, v):
        return F.rms_norm(v.float(), (self.c.d,), eps=1e-6)

    @staticmethod
    def _mix(values, keys, query):
        """Softmax over depth of q . RMSNorm(v_i) for stacked values/keys (S, B, L, d); query (d,) or (B, L, d)."""
        weights = torch.softmax((keys * query.float()).sum(-1), dim=0).to(values.dtype)
        return (weights.unsqueeze(-1) * values).sum(0)

    def _choose(self, state, step, shape):
        mode = self.dot['mode']
        if mode == 'fixed':
            return torch.full(shape, step % LIBRARY, dtype=torch.long, device=state.device), None
        if mode == 'random':
            g = torch.Generator(device='cpu').manual_seed(7919 * (step + 1))
            return torch.randint(0, LIBRARY, shape, generator=g).to(state.device), None
        logits = F.rms_norm(state.float(), (self.c.d,), eps=1e-6) @ self.dot_router[step].float().t()
        probs = torch.softmax(logits, -1)
        choice = probs.argmax(-1)
        gate = probs.gather(-1, choice.unsqueeze(-1))  # Switch: the chosen block's output is scaled by p
        if self.training and torch.is_grad_enabled():
            frac = F.one_hot(choice, LIBRARY).float().reshape(-1, LIBRARY).mean(0)
            balance = BALANCE * LIBRARY * (frac.detach() * probs.reshape(-1, LIBRARY).mean(0)).sum()
            self.aux = balance if self.aux is None else self.aux + balance
        return choice, gate

    def _attention(self, inputs, onehot, pos, seg):
        """inputs[b] is the (B, L, d) input of library block b; returns the chosen block's output per token."""
        out = 0
        for b, block in enumerate(self.blocks):
            out = out + onehot[..., b:b + 1] * block.self_attn(block.input_layernorm(inputs[b]), pos, seg)
        return out

    def _mlp(self, u, choice):
        """Sparse dispatch: each token's (B, L, d) input u goes through its chosen block's norm + MLP only."""
        shape = u.shape
        flat, idx = u.reshape(-1, shape[-1]), choice.reshape(-1)
        perm = torch.argsort(idx, stable=True)
        inverse = torch.argsort(perm, stable=True)
        counts = torch.bincount(idx, minlength=LIBRARY).tolist()
        parts = torch.split(permute(flat, perm, inverse), counts)
        outs = [self.blocks[b].mlp(self.blocks[b].post_attention_layernorm(p)) if len(p) else p
                for b, p in enumerate(parts)]
        return permute(torch.cat(outs), inverse, perm).reshape(shape)

    # ----- forward -------------------------------------------------------------------------------
    def hidden(self, x, path, *, alpha=1., donor=None, skip=None):
        from .embedding_anchor_runtime import anchor_device
        if alpha != 1. or donor is not None or skip is not None:
            raise ValueError('DOT takes no anchor, donor or skip options')
        if x.device.type != anchor_device() or self.embed.weight.device.type != anchor_device():
            raise RuntimeError('Explicit matching anchor device scope required')
        if tuple(path) != tuple(range(LIBRARY)):
            raise ValueError('DOT arms take the native library path only')
        pos = torch.arange(x.shape[1], device=x.device).expand_as(x)
        seg = torch.zeros_like(x)
        self.aux = None
        e = self.embed(x)
        multi = self.dot['residual'] == 'multi'
        sources, keys, h = [e], [self._key(e)], e  # RMSNorm keys are computed once per source
        for step in range(self.dot['steps']):
            if multi:
                V, K = torch.stack(sources), torch.stack(keys)
                state = self._mix(V, K, self.attnres_router[step])
            else:
                state = h
            choice, gate = self._choose(state, step, x.shape)
            if self.trace is not None:
                self.trace.append(choice.detach())
            onehot = F.one_hot(choice, LIBRARY).to(e.dtype)
            inputs = [self._mix(V, K, self.attnres_query[2 * b]) for b in range(LIBRARY)] if multi else [h] * LIBRARY
            a = self._attention(inputs, onehot, pos, seg)
            if gate is not None:
                a = a * gate.to(a.dtype)
            if multi:
                sources.append(a)
                keys.append(self._key(a))
                query = (onehot.unsqueeze(-1) * self.attnres_query[1:2 * LIBRARY:2].to(e.dtype)).sum(-2)
                u = self._mix(torch.stack(sources), torch.stack(keys), query)  # chosen block's MLP query per token
            else:
                h = h + a
                u = h
            m = self._mlp(u, choice)
            if gate is not None:
                m = m * gate.to(m.dtype)
            if multi:
                sources.append(m)
                keys.append(self._key(m))
            else:
                h = h + m
        return self._mix(torch.stack(sources), torch.stack(keys), self.attnres_query[2 * LIBRARY]) if multi else h
