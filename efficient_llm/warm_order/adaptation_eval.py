"""CPU integration of paired-model evaluation and lossless score replay.

Caller supplies frozen models and authenticated rows. This is not a production
loader/admission. Exports scores only, never weights or hidden activations.
"""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .adaptation_analysis import CANONICAL, SWAPPED, _frozen_cpu, weight_cells
from .adaptation_components import component_crossovers, score_crossovers
from .data import sha
from .horizon_train import write_json
from .layer_dependence import paired_scores
from .adaptation_runtime import measurement_policy


def _summary(raw, labels):
    n = len(raw['row_ids'])
    if raw['row_ids'].ndim != 1 or raw['row_ids'].dtype.kind not in 'iu' or len(np.unique(raw['row_ids'])) != n or not n:
        raise ValueError('Invalid evaluation row identities')
    expected = {'row_ids'}
    rows, losses = [], {}
    counts = None
    for i, label in enumerate(labels):
        fields = {}
        for key in ('nll_sum', 'reference_nll_sum', 'kl_reference_to_intervention_sum', 'agreement_sum', 'token_count'):
            name = f'{i}/{key}'; expected.add(name); value = raw[name]
            if value.shape != (n,) or not np.isfinite(value).all():
                raise ValueError('Invalid score shape/values')
            fields[key] = value
        c = fields['token_count']
        if c.dtype.kind not in 'iu' or (c <= 0).any():
            raise ValueError('Invalid token denominators')
        if counts is None: counts = c
        elif not np.array_equal(c, counts): raise ValueError('Unpaired token denominators')
        if (fields['nll_sum'] < 0).any() or (fields['agreement_sum'] < 0).any() or (fields['agreement_sum'] > c).any():
            raise ValueError('Invalid loss/agreement')
        den = int(c.sum()); losses[label] = fields['nll_sum']
        rows.append(dict(case=label, tokens=den, nll=float(fields['nll_sum'].sum()/den),
                         kl_to_sameweight_reference=float(fields['kl_reference_to_intervention_sum'].sum()/den)))
    if set(raw) != expected or len(set(labels)) != len(labels):
        raise ValueError('Unexpected or duplicated result members')
    den = int(counts.sum())
    contrasts = {'remaining_gap_Bswap_minus_Acanonical': float((losses['B/swap']-losses['A/canonical']).sum()/den)}
    for family in ('weights/A', 'weights/B', 'component/attention/forward',
                   'component/attention/reverse', 'component/ffn/forward', 'component/ffn/reverse'):
        a, b, c, d = (losses[family+'/'+cell] for cell in ('00', '10', '01', '11'))
        effects = dict(first_recipient=a-b, first_donor=c-d, second_recipient=a-c,
                       second_donor=b-d, joint=a-d, interaction=(c-d)-(a-b))
        for name, value in effects.items(): contrasts[family+'/'+name] = float(value.sum()/den)
    return dict(cases=rows, contrasts_nat_per_token=contrasts,
                sequences=n, tokens_per_case=den, production_ready=False,
                scope='Paired evaluation; no document-independence or causal percentage claim',
                kl_scope='Each case uses its own weights/readout reference; not cross-model KL')


@torch.no_grad()
def collect(models, tokens, row_ids):
    """One snapshot panel: parent/A/B orders, weight cells, parent component cells.

    Target block2 fixed by protocol, both intervention directions. Does not select
    favorable components from measured losses. Future snapshot callers must pin
    parent/phase provenance and batch-stream this evaluator for production scale.
    """
    if set(models) != {'parent', 'A', 'B'}:
        raise ValueError('Parent and both continuation branches required')
    for model in models.values(): _frozen_cpu(model)
    if any(asdict(m.c) != asdict(models['parent'].c) for m in models.values()):
        raise ValueError('Model config mismatch')
    ids = np.asarray(row_ids)
    if ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(ids) != len(tokens) or len(np.unique(ids)) != len(ids):
        raise ValueError('Unique paired integer rows required')
    raw, labels, parity = {'row_ids': ids.copy()}, [], {}

    def append(label, score):
        i = len(labels); labels.append(label)
        for key, value in score.items(): raw[f'{i}/{key}'] = value.detach().cpu().numpy().copy()

    for name in ('parent', 'A', 'B'):
        model = models[name]; reference = model.hidden(tokens, CANONICAL)
        for order, path in [('canonical', CANONICAL), ('swap', SWAPPED)]:
            h = model.hidden(tokens, path)
            append(name+'/'+order, paired_scores(model, tokens, h, reference))
    for branch in ('A', 'B'):
        for cell, model, sources in weight_cells(models['parent'], models[branch]):
            h = model.hidden(tokens, SWAPPED)
            append('weights/'+branch+'/'+cell, paired_scores(model, tokens, h, h))
    for kind in ('attention', 'ffn'):
        for direction, donor, recipient in [('forward', CANONICAL, SWAPPED), ('reverse', SWAPPED, CANONICAL)]:
            result = component_crossovers(models['parent'], tokens, 2, kind,
                                          donor_path=donor, recipient_path=recipient)
            scored = score_crossovers(models['parent'], tokens, result)
            family = 'component/'+kind+'/'+direction
            parity[family] = result['parity_max_abs']
            for cell in ('00', '10', '01', '11'): append(family+'/'+cell, scored['cells'][cell])
    return raw, labels, parity


def write_results(raw, labels, parity, config, out, provenance=None):
    out = Path(out)
    if out.exists() and any(out.iterdir()): raise ValueError('Fresh evaluation output required')
    report = _summary(raw, labels)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/'scores.npz', **raw)
    write_json(out/'REPORT.json', report)
    write_json(out/'manifest.json', dict(schema=1, cases=labels, config=config,
        parity_max_abs=parity, files={f: sha(out/f) for f in ('scores.npz', 'REPORT.json')},
        weights_exported=False, activations_exported=False, production_ready=False,
        checkpoint_provenance_verified=False, row_provenance_verified=False,
        provenance=provenance, measurement_runtime=measurement_policy()))
    return report


def evaluate(models, tokens, row_ids, out):
    if Path(out).exists() and any(Path(out).iterdir()): raise ValueError('Fresh evaluation output required')
    raw, labels, parity = collect(models, tokens, row_ids)
    return write_results(raw, labels, parity, asdict(models['parent'].c), out)


def replay(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest['schema'] != 1 or set(manifest['files']) != {'scores.npz', 'REPORT.json'}:
        raise ValueError('Unexpected result schema')
    for name, expected in manifest['files'].items():
        if sha(directory/name) != expected: raise ValueError('Result hash mismatch: '+name)
    with np.load(directory/'scores.npz', allow_pickle=False) as archive:
        raw = dict(archive)
    result = _summary(raw, manifest['cases'])
    if result != json.loads((directory/'REPORT.json').read_text()):
        raise ValueError('Raw-score/report replay mismatch')
    return dict(status='PASS', cases=len(manifest['cases']), sequences=result['sequences'],
                scope='Local score replay only; no actual-checkpoint or production admission')
