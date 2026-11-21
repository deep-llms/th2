"""CPU-only, same-example component interventions in the original model.

Four cells use one fixed recipient residual/suffix. Bit0 means recipient,
bit1 donor. Attention cells cross pattern/value; FFN cells cross gate/up.
No learned alignment, parameter edits, router or production admission.
"""
import math

import torch
from torch.nn import functional as F

from .adaptation_analysis import CANONICAL, SWAPPED, native_trace, _frozen_cpu
from .layer_dependence import paired_scores

# Fixed before qualification; GPU/BF16 needs its own admission and tolerances.
ATOL = 2e-6
RTOL = 1e-5


def _attention_parts(attention, x):
    b, t, _ = x.shape
    c = attention.c
    q = attention.q_norm(attention.q_proj(x).view(b, t, c.heads, c.head_dim))
    k = attention.k_norm(attention.k_proj(x).view(b, t, c.kv_heads, c.head_dim))
    v = attention.v_proj(x).view(b, t, c.kv_heads, c.head_dim)
    pos = torch.arange(t, device=x.device).expand(b, t)
    angle = pos.float()[..., None]*attention.inv
    angle = torch.cat([angle, angle], -1)[:, :, None, :]
    cos, sin = angle.cos().to(q.dtype), angle.sin().to(q.dtype)

    def rotate(z):
        return z*cos + torch.cat([-z[..., c.head_dim//2:], z[..., :c.head_dim//2]], -1)*sin

    q, k = rotate(q).transpose(1, 2), rotate(k).transpose(1, 2)
    k = k.repeat_interleave(c.heads//c.kv_heads, 1)
    v = v.transpose(1, 2).repeat_interleave(c.heads//c.kv_heads, 1)
    causal = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
    logits = (q@k.transpose(-1, -2))/math.sqrt(c.head_dim)
    pattern = logits.masked_fill(~causal, -torch.inf).softmax(-1)
    return dict(q=q, k=k, pattern=pattern, value=v)


def _attention_output(attention, pattern, value):
    b, _, t, _ = value.shape
    return attention.o_proj((pattern@value).transpose(1, 2).reshape(b, t, -1))


def _ffn_parts(mlp, x):
    return dict(gate=F.silu(mlp.gate_proj(x)), up=mlp.up_proj(x))


def _parity(candidate, reference, label):
    if not torch.isfinite(candidate).all() or not torch.allclose(candidate, reference, atol=ATOL, rtol=RTOL):
        raise ValueError('CPU component parity failed: '+label)
    return float((candidate-reference).abs().max())


@torch.no_grad()
def component_crossovers(model, tokens, target, kind, *, donor_path=CANONICAL, recipient_path=SWAPPED):
    """Return four hidden-state interventions, true updates and reference runs.

    Same fixed block weights and token sequence in all cells. For attention,
    the recipient FFN is naturally recomputed after the changed attention update;
    for FFN, the recipient attention remains unchanged. Donor-donor inside the
    recipient is NOT called a complete donor forward. All activations stay local.
    """
    _frozen_cpu(model)
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('CPU FP32 qualification only')
    if type(target) is not int or target not in range(4) or kind not in ('attention', 'ffn'):
        raise ValueError('Invalid target/component')
    recipient_hidden, recipient_trace = native_trace(model, tokens, recipient_path)
    donor_hidden, donor_trace = native_trace(model, tokens, donor_path)
    name = f'middle.{target}'
    records = [next(r for r in trace if r['block']==name) for trace in (recipient_trace, donor_trace)]
    block = model.middle[target]
    if kind == 'attention':
        module = block.self_attn
        parts = [_attention_parts(module, r['attention_input_normalized']) for r in records]
        def compose(i, j):
            return _attention_output(module, parts[i]['pattern'], parts[j]['value'])
        native_key = 'attention_update'
        labels = ('pattern', 'value')
    else:
        module = block.mlp
        parts = [_ffn_parts(module, r['ffn_input_normalized']) for r in records]
        def compose(i, j):
            return module.down_proj(parts[i]['gate']*parts[j]['up'])
        native_key = 'ffn_update'
        labels = ('gate', 'up')
    parity = {which: _parity(compose(i, i), records[i][native_key], which)
              for i, which in enumerate(('recipient_update', 'donor_update'))}
    cells = {}
    for i, j in ((0, 0), (1, 0), (0, 1), (1, 1)):
        update = compose(i, j)
        calls = []
        def replace(module, args, output):
            calls.append(1)
            if output.shape != update.shape:
                raise ValueError('Intervention shape mismatch')
            return update
        handle = module.register_forward_hook(replace)
        try:
            hidden = model.hidden(tokens, recipient_path)
        finally:
            handle.remove()
        if len(calls) != 1 or not torch.isfinite(hidden).all():
            raise ValueError('Invalid intervention execution')
        cells[f'{i}{j}'] = dict(hidden=hidden, update=update)
    parity['recipient_hidden'] = _parity(cells['00']['hidden'], recipient_hidden, 'recipient hidden')
    return dict(cells=cells, recipient_hidden=recipient_hidden, donor_hidden=donor_hidden,
                parts=parts, parity_max_abs=parity,
                metadata=dict(target_block=name, kind=kind, bits=list(labels),
                              zero='recipient', one='donor', recipient_path=list(recipient_path),
                              donor_path=list(donor_path), fixed_weights=True,
                              fixed_recipient_residual=True, fixed_recipient_suffix_weights=True,
                              cpu_only=True, atol=ATOL, rtol=RTOL))


@torch.no_grad()
def score_crossovers(model, tokens, result):
    """Small per-sequence score arrays, retaining exact token denominators.

    Reference is the unmodified recipient. KL fidelity and NLL quality remain
    separate; effects are unnormalized sums, not fractions of recovered damage.
    This output is not a document-independent uncertainty estimate.
    """
    _frozen_cpu(model)
    if set(result['cells']) != {'00', '10', '01', '11'}:
        raise ValueError('Expected all four intervention cells')
    reference = result['recipient_hidden']
    scores = {key: paired_scores(model, tokens, value['hidden'], reference)
              for key, value in result['cells'].items()}
    a, b, c, d = (scores[key]['nll_sum'] for key in ('00', '10', '01', '11'))
    effects = dict(first_with_recipient_second=a-b,
                   first_with_donor_second=c-d,
                   second_with_recipient_first=a-c,
                   second_with_donor_first=b-d,
                   interaction=(c-d)-(a-b), joint=a-d)
    return dict(cells=scores, effects_nll_sum=effects,
                recipient=paired_scores(model, tokens, reference, reference),
                donor=paired_scores(model, tokens, result['donor_hidden'], reference),
                metadata=result['metadata'], parity_max_abs=result['parity_max_abs'])
