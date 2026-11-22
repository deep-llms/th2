"""CPU score collection and portable replay; production checkpoint joins pending."""
import json
from pathlib import Path
import random

import numpy as np
import torch

from .attention_shift import fingerprint
from .data import sha
from .embedding_anchor import TRAIN_PATHS, HELD_OUT_PATHS
from .embedding_anchor_data import BoundRows
from .embedding_anchor_train import CPUStreamTrainer


CASES = {'path' + ''.join(map(str, p)): dict(path=list(p), skip=None)
         for p in sorted(TRAIN_PATHS + HELD_OUT_PATHS)}
for p in ((0, 1, 2, 3), (0, 2, 1, 3)):
    for block in p:
        CASES['path' + ''.join(map(str, p)) + '-skip' + str(block)] = dict(path=list(p), skip=block)
for block in range(4):
    CASES['single' + str(block)] = dict(path=[block], skip=None)
LABELS = [arm + '/' + name for arm in ('A', 'C') for name in CASES]


def summarize(raw):
    if set(raw) != set(LABELS) | {'row_ids', 'token_count'}:
        raise ValueError('Endpoint set mismatch')
    ids, counts = raw['row_ids'], raw['token_count']
    if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or not len(ids)
            or len(np.unique(ids)) != len(ids) or (ids < 0).any()
            or counts.shape != ids.shape or counts.dtype.kind not in 'iu' or (counts <= 0).any()):
        raise ValueError('Invalid row IDs/denominators')
    for label in LABELS:
        v = raw[label]
        if v.shape != ids.shape or not np.isfinite(v).all() or (v < 0).any():
            raise ValueError('Invalid per-row NLL sums')
    den = int(counts.sum())
    report = dict(rows=len(ids), tokens_per_case=den,
                  nll={k: float(raw[k].sum() / den) for k in LABELS}, paths={}, skip={})
    for family, paths in [('seen_even', TRAIN_PATHS), ('unseen_full_order_odd', HELD_OUT_PATHS)]:
        names = ['path' + ''.join(map(str, p)) for p in paths]
        means = {}
        for arm in ('A', 'C'):
            matrix = np.stack([raw[arm + '/' + name] for name in names])
            means[arm] = matrix.mean(0)
            report['paths'][arm + '/' + family] = dict(
                mean_nll=float(matrix.sum() / (len(paths) * den)),
                worst_path_nll=float(matrix.sum(1).max() / den),
                perrow_path_range_quantiles=np.quantile(np.ptp(matrix / counts, axis=0), [.1, .5, .9]).tolist())
        differences = means['C'] - means['A']
        report['paths']['A_gain_over_C/' + family] = dict(
            nll_gain=float(differences.sum() / den),
            perrow_gain_quantiles=np.quantile(differences / counts, [.1, .5, .9]).tolist(),
            positive_rows=int((differences > 0).sum()), negative_rows=int((differences < 0).sum()))
    for arm in ('A', 'C'):
        report['paths'][arm + '/odd_minus_even_nll'] = (
            report['paths'][arm + '/unseen_full_order_odd']['mean_nll']
            - report['paths'][arm + '/seen_even']['mean_nll'])
        for name in CASES:
            if '-skip' not in name: continue
            diff = raw[arm + '/' + name] - raw[arm + '/' + name.split('-')[0]]
            report['skip'][arm + '/' + name] = dict(
                loss_cost=float(diff.sum() / den),
                perrow_cost_quantiles=np.quantile(diff / counts, [.1, .5, .9]).tolist(),
                positive_rows=int((diff > 0).sum()), negative_rows=int((diff < 0).sum()))
    report['scope'] = 'Exploratory paired paths; conditional skip costs, no causal percentages or freshness claim'
    return report


def evaluate_fixture(models, data, out, *, ids=None, batch=2):
    """Models are in-memory fixtures, NOT claimed to be admitted terminal fits."""
    if (set(models) != {'A', 'C'} or not isinstance(data, BoundRows)
            or not data.binding['fixture_only'] or type(batch) is not int or batch <= 0):
        raise ValueError('Two CPU fixture models and authenticated fixture data required')
    return _evaluate(models, data, out, ids=ids, batch=batch, proofs=None)


def evaluate_checkpoints(runs, warmup, config, data, out, *, seed=0, step=1920, ids=None, batch=2):
    """Authenticate common ancestor, both fullstates and data before scoring.

Alternate registered snapshots/subsets are fixture checks only. Production
measurement requires terminal1920 and all512 rows. CUDA remains unsupported.
"""
    if (set(runs) != {'A', 'C'} or type(batch) is not int or batch <= 0
            or not isinstance(data, BoundRows) or step not in (1152, 1344, 1536, 1920)):
        raise ValueError('Invalid paired evaluation registration')
    if not data.binding['fixture_only'] and (step != 1920 or ids is not None):
        raise ValueError('Production measurement must use terminal1920/all512rows')
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            ancestor = CPUStreamTrainer.restore(warmup['path'], warmup['sha256'], config, data, seed=seed)
            if ancestor.arm != 'shared' or ancestor.step != 768:
                raise ValueError('Expected common warm-up768 ancestor')
            ancestor_trace = ancestor.trace
            ancestor_recipe = ancestor.recipe
            del ancestor
            models, arms = {}, {}
            for arm in ('A', 'C'):
                t = CPUStreamTrainer.restore(runs[arm]['path'], runs[arm]['sha256'], config, data, seed=seed)
                if (t.arm != arm or t.step != step or t.fork_sha256 != warmup['sha256']
                        or t.recipe != ancestor_recipe or t.trace[:768] != ancestor_trace):
                    raise ValueError('Checkpoint arm/step/ancestor mismatch')
                models[arm] = t.model
                arms[arm] = dict(checkpoint=dict(runs[arm]), step=t.step, arm=arm,
                                 fork_sha256=t.fork_sha256, model_fingerprint=fingerprint(t.model))
                del t
            proofs = dict(warmup=dict(warmup), warmup_step=768, step=step, arms=arms,
                          data_binding=data.binding, model_config=ancestor_recipe['model'],
                          seed=seed, source=ancestor_recipe['source'])
            return _evaluate(models, data, out, ids=ids, batch=batch, proofs=proofs)
    finally:
        random.setstate(python_rng); np.random.set_state(numpy_rng)


def _evaluate(models, data, out, *, ids, batch, proofs):
    if models['A'].c != models['C'].c or models['A'].c.length != data.length:
        raise ValueError('Model/data shape mismatch')
    ids = data.measurement_ids if ids is None else np.asarray(ids)
    if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or not len(ids)
            or len(np.unique(ids)) != len(ids) or not np.isin(ids, data.measurement_ids).all()):
        raise ValueError('Only unique bound measurement rows may be scored')
    out = Path(out)
    if out.exists(): raise ValueError('Fresh output directory required')
    before = {arm: fingerprint(m) for arm, m in models.items()}
    modes = {arm: m.training for arm, m in models.items()}
    raw = {label: [] for label in LABELS}
    try:
        for m in models.values(): m.eval()
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            for start in range(0, len(ids), batch):
                x = data.batch(ids[start:start + batch])
                for arm, model in models.items():
                    for name, case in CASES.items():
                        scores = model.token_losses(x, case['path'], skip=case['skip'], alpha=.5 if arm == 'A' else 1.)
                        raw[arm + '/' + name].append(scores.double().sum(1).numpy())
    finally:
        for arm, m in models.items(): m.train(modes[arm])
    if before != {arm: fingerprint(m) for arm, m in models.items()}:
        raise ValueError('Evaluation changed model weights/buffers')
    raw = {k: np.concatenate(v) for k, v in raw.items()}
    raw.update(row_ids=ids.copy(), token_count=np.full(len(ids), data.length - models['A'].c.prefix, dtype=np.int64))
    report = summarize(raw)
    out.mkdir(parents=True)
    np.savez_compressed(out / 'scores.npz', **raw)
    (out / 'REPORT.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    meta = dict(schema=2, cases=CASES, labels=LABELS, fixture_only=data.binding['fixture_only'],
                checkpoint_binding_verified=proofs is not None, checkpoint_proofs=proofs,
                data_binding=data.binding, model_fingerprints=before, row_ids=ids.tolist(),
                tokens_per_row=data.length - models['A'].c.prefix,
                files={n: sha(out / n) for n in ('scores.npz', 'REPORT.json')})
    (out / 'manifest.json').write_text(json.dumps(meta, indent=2) + '\n')
    replay(out)
    return report


def replay(out):
    out = Path(out)
    meta = json.loads((out / 'manifest.json').read_text())
    if (meta['cases'] != CASES or meta['labels'] != LABELS
            or meta['fixture_only'] != meta['data_binding']['fixture_only']
            or set(meta['files']) != {'scores.npz', 'REPORT.json'}):
        raise ValueError('Evaluation registration mismatch')
    proofs = meta['checkpoint_proofs']
    if meta['checkpoint_binding_verified']:
        if (not proofs or set(proofs['arms']) != {'A', 'C'} or proofs['warmup_step'] != 768
                or proofs['data_binding'] != meta['data_binding']
                or proofs['step'] not in (1152, 1344, 1536, 1920)
                or proofs['seed'] != meta['data_binding']['seed']
                or proofs['model_config']['length'] - proofs['model_config']['prefix'] != meta['tokens_per_row']):
            raise ValueError('Checkpoint/data registration mismatch')
        for arm in ('A', 'C'):
            p = proofs['arms'][arm]
            if (p['arm'] != arm or p['step'] != proofs['step']
                    or p['fork_sha256'] != proofs['warmup']['sha256']
                    or p['model_fingerprint'] != meta['model_fingerprints'][arm]):
                raise ValueError('Checkpoint ancestor/arm join mismatch')
        if not meta['fixture_only'] and (proofs['step'] != 1920 or len(meta['row_ids']) != 512):
            raise ValueError('Incomplete production endpoints')
    elif proofs is not None or not meta['fixture_only']:
        raise ValueError('Missing checkpoint binding')
    for name, digest in meta['files'].items():
        if sha(out / name) != digest: raise ValueError('Evaluation file hash mismatch')
    with np.load(out / 'scores.npz', allow_pickle=False) as f:
        raw = {k: f[k] for k in f.files}
    if (raw['row_ids'].tolist() != meta['row_ids']
            or not (raw['token_count'] == meta['tokens_per_row']).all()
            or summarize(raw) != json.loads((out / 'REPORT.json').read_text())):
        raise ValueError('Raw score replay mismatch')
    return dict(status='PASS', cases_per_arm=len(CASES), rows=len(raw['row_ids']),
                fixture_only=meta['fixture_only'], checkpoint_binding_verified=meta['checkpoint_binding_verified'])
