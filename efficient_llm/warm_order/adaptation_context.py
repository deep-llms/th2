"""Frozen input × block × reader crossover, with fixed path and block identity.

Qualification primitives only: no loader, training, reservation or GPU admission.
Crossed prefix/readout sources deliberately break the native shared-weight use;
reader effects therefore include the readout, not just downstream blocks.
"""
from dataclasses import asdict
from itertools import product

import torch

from .adaptation_analysis import _frozen_cpu, SWAPPED
from .adaptation_runtime import measurement_policy
from .layer_dependence import _validate, paired_scores


@torch.no_grad()
def context_crossovers(parent, adapted, tokens, target=2, path=SWAPPED):
    models = (parent, adapted)
    for model in models:
        _frozen_cpu(model)
        _validate(model, tokens)
    if asdict(parent.c) != asdict(adapted.c):
        raise ValueError('Matching model configurations required')
    if (not isinstance(path, tuple) or len(path) != 4 or
            any(type(i) is not int for i in path) or set(path) != set(range(4))):
        raise ValueError('Fixed full permutation required')
    if type(target) is not int or target not in path:
        raise ValueError('Explicit target block identity required')
    cut = path.index(target)
    pos, seg = _validate(parent, tokens)
    inputs = []
    for model in models:
        h = model.stem(model.embed(tokens), pos, seg)
        for index in path[:cut]:
            h = model.middle[index](h, pos, seg)
        inputs.append(h)
    outputs = {(i, b): models[b].middle[target](inputs[i], pos, seg)
               for i, b in product(range(2), repeat=2)}

    def finish(h, reader):
        model = models[reader]
        for index in path[cut+1:]:
            h = model.middle[index](h, pos, seg)
        return model.norm(model.coda(h, pos, seg))

    cells = {f'{i}{b}{r}': finish(outputs[i, b], r)
             for i, b, r in product(range(2), repeat=3)}
    cells.update({f'{i}S{r}': finish(inputs[i], r) for i, r in product(range(2), repeat=2)})
    policy = measurement_policy()
    parity = {}
    for bit, model in enumerate(models):
        native = model.hidden(tokens, path)
        end = cells[str(bit)*3]
        torch.testing.assert_close(end, native, atol=policy['component_atol'], rtol=policy['component_rtol'])
        parity[str(bit)] = float((end-native).abs().max())
    if any(not torch.isfinite(h).all() for h in cells.values()):
        raise ValueError('Nonfinite crossover activation')
    return dict(cells=cells, inputs=inputs, block_outputs=outputs,
                target=target, path=path, target_slot=cut+1, parity_max_abs=parity,
                bit_order=['prefix_input_source', 'target_block_weights', 'suffix_and_readout_source'],
                skip_label='S', runtime=policy,
                interpretation='Conditional hybrid interventions; not unique causal shares or layer knowledge extraction')


@torch.no_grad()
def score_context(parent, adapted, tokens, result):
    """Same rows/masks; KL reference is native under the same reader weights."""
    scores = {}
    for key, hidden in result['cells'].items():
        reader = int(key[2]);model = (parent, adapted)[reader]
        scores[key] = paired_scores(model, tokens, hidden, result['cells'][str(reader)*3])
    return dict(cells=scores, effects_nll_sum=context_effects({k: v['nll_sum'] for k, v in scores.items()}),
                kl_scope='Same suffix/readout native reference; do not compare KL across readers')


def context_effects(loss):
    expected = {f'{i}{b}{r}' for i,b,r in product(range(2), repeat=3)}
    expected |= {f'{i}S{r}' for i,r in product(range(2), repeat=2)}
    if set(loss) != expected:
        raise ValueError('Eight crossed cells and four matched skips required')
    sample = loss['000']
    if not isinstance(sample, torch.Tensor):
        raise ValueError('Tensor losses required')
    if any(not isinstance(v, torch.Tensor) or v.shape != sample.shape or not torch.isfinite(v).all()
           for v in loss.values()):
        raise ValueError('Finite matched losses required')
    out = {}
    for r in range(2):
        for b in range(2):
            out[f'input/block{b}/reader{r}'] = loss[f'0{b}{r}']-loss[f'1{b}{r}']
        for i in range(2):
            out[f'block/input{i}/reader{r}'] = loss[f'{i}0{r}']-loss[f'{i}1{r}']
            for b in range(2):
                out[f'usefulness/input{i}/block{b}/reader{r}'] = loss[f'{i}S{r}']-loss[f'{i}{b}{r}']
        out[f'input_block_interaction/reader{r}'] = (
            loss[f'10{r}']-loss[f'11{r}']-loss[f'00{r}']+loss[f'01{r}'])
    for i,b in product(range(2), repeat=2):
        out[f'reader/input{i}/block{b}'] = loss[f'{i}{b}0']-loss[f'{i}{b}1']
    out['total'] = loss['000']-loss['111']
    return out
