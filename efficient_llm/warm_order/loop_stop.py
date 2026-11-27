"""Per-token STOP over loop passes with copy-through state semantics.

Design: docs/LOOP_STOP_20261123.md. A token that halts after pass s keeps, for every
later pass, (a) its loop state, (b) the exact input it last fed to each core block
sublayer, so the keys/values other tokens read from it equal a cache taken at pass s,
and (c) for AttnRes arms, depth sources produced after s are masked out of its mixes.
Compute is logical: halted tokens still flow through dense kernels here, but their
results are discarded; accounting counts only active-token block applications.
Halting decisions use only the token's own causal state, never targets.
"""
import torch
from torch import nn
from torch.nn import functional as F


class HaltHead(nn.Module):
    """q_k = sigmoid(w . RMSNorm(state) + b_k): probability of stopping after pass k."""

    def __init__(self, d, kmax):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(d))
        self.b = nn.Parameter(torch.zeros(kmax))
        self.kmax = kmax

    def forward(self, state, k):
        z = F.rms_norm(state.float(), (state.shape[-1],), eps=1e-6)
        return torch.sigmoid((z * self.w.float()).sum(-1) + self.b[k - 1].float())


def _keep(active, new, old):
    return torch.where(active.unsqueeze(-1), new, old)


def _block(block, x, pos, seg, active, frozen, key):
    """Run a native Block where halted tokens feed their frozen inputs (copy-through)."""
    x_in = x if frozen.get(key) is None else _keep(active, x, frozen[key])
    frozen[key] = x_in
    z_new = block.attention(x_in, pos, seg)
    z_in = z_new if frozen.get(key+('ffn',)) is None else _keep(active, z_new, frozen[key+('ffn',)])
    frozen[key+('ffn',)] = z_in
    return z_in + block.mlp(block.post_attention_layernorm(z_in))


def _mix(model, sources, valid, index):
    values = torch.stack(sources, dim=0)
    keys = F.rms_norm(values.float(), (model.c.d,), eps=1e-6)
    logits = (keys * model.attnres_query[index].float()).sum(-1)
    logits = logits.masked_fill(~torch.stack(valid, dim=0), float('-inf'))
    weights = torch.softmax(logits, dim=0).to(values.dtype)
    return (weights.unsqueeze(-1) * values).sum(0)


def halted_hidden(model, x, path, kmax, decide):
    """Hidden states under per-token halting.

    decide(k, state) -> bool (B,T): tokens wishing to stop after pass k (only active ones
    are affected); every token stops at kmax. Returns (hidden, passes_used[B,T]).
    """
    spec = model.spec
    if not spec['loop']:
        raise ValueError('STOP requires a loop arm')
    model._guard(x)
    model.passes = kmax
    path = model._check_path(path)
    pos = torch.arange(x.shape[1], device=x.device).expand_as(x)
    seg = torch.zeros_like(x)
    e = model.embed(x)
    active = torch.ones(x.shape, dtype=torch.bool, device=x.device)
    used = torch.zeros(x.shape, dtype=torch.long, device=x.device)
    frozen = {}
    core = path[1:3]
    if spec['mix'] == 'attnres':
        everyone = torch.ones_like(active)
        sources, valid = [e], [everyone]

        def sub_apply(identity, j, x_in_fn, key, mask):
            block = model.blocks[identity]
            x_new = _mix(model, sources, valid, 2 * identity + j)
            x_in = x_new if frozen.get(key) is None else _keep(mask, x_new, frozen[key])
            frozen[key] = x_in
            if j == 0:
                out = block.self_attn(block.input_layernorm(x_in), pos, seg)
            else:
                out = block.mlp(block.post_attention_layernorm(x_in))
            sources.append(out)
            valid.append(mask.clone())

        for j in (0, 1):
            sub_apply(path[0], j, None, ('pre', j), everyone)
        for k in range(1, kmax + 1):
            for identity in core:
                for j in (0, 1):
                    sub_apply(identity, j, None, ('core', identity, j), active)
            used = used + active.long()
            state = _mix(model, sources, valid, 2 * model.spec['blocks'])  # readout-view of current state
            if k < kmax:
                active = active & ~decide(k, state)
        for j in (0, 1):
            sub_apply(path[3], j, None, ('coda', j), everyone)
        return _mix(model, sources, valid, 2 * model.spec['blocks']), used
    prelude = model.blocks[path[0]](e, pos, seg)
    if spec['loop'] == 'gated':
        message = prelude
        for k in range(1, kmax + 1):
            h = prelude if k == 1 else prelude + model.message_gate * (message - prelude)
            for identity in core:
                h = _block(model.blocks[identity], h, pos, seg, active, frozen, ('core', identity))
            message = _keep(active, h, message)
            used = used + active.long()
            if k < kmax:
                active = active & ~decide(k, message)
        return model.blocks[path[3]](message, pos, seg), used
    h = prelude
    previous = ()
    for k in range(1, kmax + 1):
        current = []
        state = h
        for identity in core:
            if spec['loop'] == 'memory' and model.memory_enabled:
                state = model.read_memory(state, (e, prelude, *previous), identity)
            state = _block(model.blocks[identity], state, pos, seg, active, frozen, ('core', identity))
            current.append(state)
        h = _keep(active, state, h)
        previous = tuple(current)
        used = used + active.long()
        if k < kmax:
            active = active & ~decide(k, h)
    return model.blocks[path[3]](h, pos, seg), used


def halting_distribution(q):
    """q: (K, ...) stop probabilities after pass k (last forced to 1). Returns p_k (K, ...)."""
    q = torch.cat([q[:-1], torch.ones_like(q[-1:])], dim=0)
    survive = torch.cumprod(torch.cat([torch.ones_like(q[:1]), 1 - q[:-1]], dim=0), dim=0)
    return survive * q


def ponder_loss(q, nll, lam):
    """Expected per-token NLL plus lam * expected passes; q,nll: (K, N)."""
    p = halting_distribution(q)
    passes = torch.arange(1, q.shape[0] + 1, device=q.device, dtype=p.dtype).unsqueeze(-1)
    return (p * nll).sum(0).mean() + lam * (p * passes).sum(0).mean(), (p * passes).sum(0).mean()
