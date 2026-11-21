"""Local CPU qualification of paired-adaptation measurements.

No checkpoint loader, trainer, runner or production admission. Uses the exact
existing Model forward; captures true branch outputs, not residual differences.
"""
import copy
from dataclasses import asdict

import torch

from .layer_dependence import _validate
from .model import Model
from .adaptation_runtime import measurement_device

CANONICAL = (0, 1, 2, 3)
SWAPPED = (0, 2, 1, 3)
PAIR = (1, 2)


def _frozen_cpu(model):
    if type(model) is not Model:
        raise ValueError('Expected original warm-order Model')
    if any(m.training for m in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Explicit frozen eval model required')
    if any(t.device.type != measurement_device() for t in model.state_dict().values()):
        raise ValueError('Model device differs from explicit measurement runtime')
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('Measurement requires FP32 model parameters')


@torch.no_grad()
def native_trace(model, tokens, path):
    """Capture exact native block inputs/updates with identity AND slot labels.

    Returned activations belong to this call; no persistent hooks or tensor
    references into parameters are exposed. Only full permutations are allowed.
    """
    _frozen_cpu(model)
    _validate(model, tokens)
    if (not isinstance(path, tuple) or len(path) != 4 or
            any(type(i) is not int for i in path) or set(path) != set(CANONICAL)):
        raise ValueError('Explicit full middle permutation required')
    records, handles = [], []
    blocks = [('stem', model.stem)] + [(f'middle.{i}', model.middle[i]) for i in path] + [('coda', model.coda)]
    try:
        for slot, (name, block) in enumerate(blocks):
            record = dict(block=name, execution_slot=slot)
            records.append(record)

            def pre(module, args, r=record):
                r['input'] = args[0].detach().clone()

            def save(key, r=record):
                def hook(module, args, output):
                    r[key] = output.detach().clone()
                return hook

            handles.append(block.register_forward_pre_hook(pre))
            handles.append(block.input_layernorm.register_forward_hook(save('attention_input_normalized')))
            handles.append(block.self_attn.register_forward_hook(save('attention_update')))
            handles.append(block.post_attention_layernorm.register_forward_hook(save('ffn_input_normalized')))
            handles.append(block.mlp.register_forward_hook(save('ffn_update')))
            handles.append(block.register_forward_hook(save('output')))
        hidden = model.hidden(tokens, path)
    finally:
        for handle in handles:
            handle.remove()
    for record in records:
        record['post_attention'] = record['input'] + record['attention_update']
    return hidden, records


def weight_cells(parent, adapted):
    """Yield 00/10/01/11 frozen models plus exact tensor-source manifests.

    First bit selects adapted pair weights, second selects adapted remainder.
    The embedding/readout remains one shared tensor. No optimizer is spliced.
    Each yielded model is independent; neither input model is modified.
    """
    _frozen_cpu(parent)
    _frozen_cpu(adapted)
    if asdict(parent.c) != asdict(adapted.c):
        raise ValueError('Model config mismatch')
    before, after = parent.state_dict(), adapted.state_dict()
    if before.keys() != after.keys():
        raise ValueError('State schema mismatch')
    for key, value in before.items():
        other = after[key]
        if value.shape != other.shape or value.dtype != other.dtype:
            raise ValueError(f'Tensor schema mismatch: {key}')
        if not torch.isfinite(value).all() or not torch.isfinite(other).all():
            raise ValueError(f'Nonfinite state: {key}')
    prefixes = tuple(f'middle.{i}.' for i in PAIR)
    for pair_bit, rest_bit in ((0, 0), (1, 0), (0, 1), (1, 1)):
        sources = {key: (pair_bit if key.startswith(prefixes) else rest_bit) for key in before}
        state = {key: (after if sources[key] else before)[key] for key in before}
        model = copy.deepcopy(parent)
        model.load_state_dict(state, strict=True)
        yield f'{pair_bit}{rest_bit}', model, sources


def loss_contrasts(cells):
    """Unnormalized paired NLL effects; last axis is NOT silently averaged.

    Inputs can be per-sequence or per-token arrays with identical shape.
    Callers retain token denominators and perform the registered aggregation.
    """
    if set(cells) != {'00', '10', '01', '11'}:
        raise ValueError('All four loss cells required')
    values = [cells[k] for k in ('00', '10', '01', '11')]
    if any(not isinstance(v, torch.Tensor) for v in values):
        raise ValueError('Tensor losses required')
    if any(v.shape != values[0].shape or not torch.isfinite(v).all() for v in values):
        raise ValueError('Loss shape/nonfinite mismatch')
    a, b, c, d = (v.detach().double() for v in values)
    return dict(pair_parent_context=a-b, pair_adapted_context=c-d,
                rest_parent_pair=a-c, rest_adapted_pair=b-d,
                interaction=(c-d)-(a-b), total_recovery=a-d)
