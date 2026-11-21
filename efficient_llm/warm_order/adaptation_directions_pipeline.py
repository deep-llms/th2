"""Bound calibration and small-result replay for the fixed V/O direction study.

Basis/moments are node artifacts. Portable replay checks scalar/score outputs;
it does not pretend to re-execute the model or recover the basis off-node.
"""
import json
from pathlib import Path

import numpy as np
import torch

from .adaptation_directions import DeltaMoments, CONTROL_SEEDS, vo_boundary, finish_update, update_cases
from .adaptation_runtime import measurement_device, measurement_policy
from .data import Rows, sha
from .horizon_train import write_json
from .layer_dependence import paired_scores

CELLS = ['base', 'full', 'mean', 'projected', 'complement', 'centered', 'affine',
         'reverse', 'radial', 'norm_restored'] + [f'{mode}/{seed}' for seed in CONTROL_SEEDS
                                                for mode in ('random', 'random_energy')]
LABELS = [f'{b}/{cell}' for b in ('A', 'B') for cell in CELLS]
FLAGS = ['exploratory', 'reused_confirmation', 'historical_overlap_unknown',
         'fixed_parent_input_reader', 'per_input_donor_required']


def read(p):
    return json.loads(Path(p).read_text())


def validate(c):
    if (c.get('schema') != 1 or c.get('domain') != 'dclm' or c.get('target') != 2 or
            c.get('path') != [0, 2, 1, 3] or c.get('rank') != 8 or
            c.get('control_seeds') != list(CONTROL_SEEDS) or
            c.get('positions') != 'all_predictor_positions_0_to_length_minus_2' or
            c.get('measurement_device') not in ('cpu', 'cuda') or c.get('flags') != FLAGS or
            type(c.get('batch_size')) is not int or c['batch_size'] <= 0):
        raise ValueError('Unregistered directional configuration')


def selections(c, length):
    validate(c)
    paths = (c['data']+'.bin', c['data']+'.starts', c['indices'])
    if {p: sha(p) for p in paths} != c['expected_inputs']:
        raise ValueError('Evaluation input identity mismatch')
    if sha(c['calibration_indices']) != c['calibration_sha256']:
        raise ValueError('Calibration identity mismatch')
    rows = Rows(c['data'], length)
    ids = [np.load(c[key], allow_pickle=False) for key in ('calibration_indices', 'indices')]
    for v in ids:
        if (v.ndim != 1 or v.dtype.kind not in 'iu' or not len(v) or len(np.unique(v)) != len(v)
                or v.min() < 0 or v.max() >= len(rows.ids)):
            raise ValueError('Unique in-range row IDs required')
    if np.intersect1d(*ids).size:
        raise ValueError('Calibration/evaluation overlap')
    return rows, ids[0], ids[1]


def check_proof(c, proof):
    if (proof.get('checkpoint_bytes_and_pairing_verified') is not True or
            proof['phase_files'] != c['phase_files'] or proof['parent_file_pins'] != c['parent_hashes'] or
            proof['phase_step'] != c['phase_step']):
        raise ValueError('Checkpoint binding mismatch')


def binding(c, proof, cal, length):
    check_proof(c, proof)
    return dict(checkpoints=proof, data_pins={p: c['expected_inputs'][p] for p in
                (c['data']+'.bin', c['data']+'.starts')},
                calibration_indices=c['calibration_indices'], calibration_sha256=c['calibration_sha256'],
                calibration_row_ids=cal.tolist(), length=length, rank=c['rank'],
                positions=c['positions'], control_seeds=c['control_seeds'], target=c['target'], path=c['path'])


def fit_summary(fitted):
    values = fitted['eigenvalues']; total = float(values.sum()); rank = fitted['actual_rank']
    return dict(requested_rank=fitted['requested_rank'], actual_rank=rank,
                observations=fitted['observations'], numerical_threshold=fitted['numerical_threshold'],
                boundary_eigengap=fitted['boundary_eigengap'], mean_squared_norm=float(fitted['mean'].square().sum()),
                centered_scatter_eigenvalues=values.tolist(), centered_scatter_trace=total,
                centered_energy_fraction=float(values[:rank].sum())/total if total > 0 else None)


@torch.no_grad()
def calibrate(models, c, proof, out):
    if measurement_device() != 'cpu':
        raise ValueError('Calibration is registered CPU-only')
    out = Path(out)
    if out.exists():
        raise ValueError('Fresh calibration directory required')
    rows, cal, _ = selections(c, models['parent'].c.length)
    provenance = binding(c, proof, cal, models['parent'].c.length)
    out.mkdir(parents=True); summaries = {}; files = {}
    for branch in ('A', 'B'):
        moments = DeltaMoments(models['parent'].c.d)
        for start in range(0, len(cal), c['batch_size']):
            x = rows.batch(cal[start:start+c['batch_size']], 'cpu')
            b = vo_boundary(models['parent'], models[branch], x)
            delta = (b['donor']-b['base'])[:, :-1].reshape(-1, models['parent'].c.d).cpu()
            moments.update(delta)
            write_json(out/'progress.json', dict(branch=branch, rows=min(start+c['batch_size'], len(cal)), total=len(cal)))
        fitted = moments.fit(c['rank'])
        np.savez_compressed(out/f'{branch}-moments.npz', **moments.arrays())
        torch.save(fitted, out/f'{branch}-basis.pt')
        summaries[branch] = fit_summary(fitted)
        for suffix in ('moments.npz', 'basis.pt'):
            name = f'{branch}-{suffix}'; files[name] = sha(out/name)
    write_json(out/'calibration.json', dict(schema=1, binding=provenance, panels=summaries,
                node_files=files, basis_exported=False, moments_exported=False))
    replay_calibration(out)
    return read(out/'calibration.json')


def replay_calibration(root):
    root = Path(root); m = read(root/'calibration.json'); fitted = {}
    expected = {f'{b}-{s}' for b in ('A', 'B') for s in ('moments.npz', 'basis.pt')}
    if set(m['node_files']) != expected:
        raise ValueError('Unexpected node artifact set')
    for name, h in m['node_files'].items():
        if sha(root/name) != h:
            raise ValueError('Calibration artifact hash mismatch')
    for b in ('A', 'B'):
        with np.load(root/f'{b}-moments.npz', allow_pickle=False) as f:
            recomputed = DeltaMoments.from_arrays({k: f[k] for k in f.files}).fit(m['binding']['rank'])
        saved = torch.load(root/f'{b}-basis.pt', map_location='cpu', weights_only=True)
        if fit_summary(saved) != m['panels'][b]:
            raise ValueError('Calibration scalar replay mismatch')
        for key in ('mean', 'eigenvalues'):
            torch.testing.assert_close(saved[key], recomputed[key], atol=1e-10, rtol=1e-9)
        if saved['actual_rank'] != recomputed['actual_rank']:
            raise ValueError('Calibration rank mismatch')
        torch.testing.assert_close(saved['basis']@saved['basis'].T,
                                   recomputed['basis']@recomputed['basis'].T, atol=1e-9, rtol=1e-8)
        if saved['observations'] != len(m['binding']['calibration_row_ids'])*(m['binding']['length']-1):
            raise ValueError('Calibration observation count mismatch')
        fitted[b] = saved
    return fitted


@torch.no_grad()
def panel(models, x, fitted, *, reference=False):
    """Stream suffixes: retain all hidden outputs only for the single-row parity gate."""
    result = {}; diagnostics = {}
    for branch in ('A', 'B'):
        boundary = vo_boundary(models['parent'], models[branch], x)
        updates = update_cases(boundary['base'], boundary['donor'], fitted[branch])
        if list(updates) != CELLS:
            raise ValueError('Directional case registration mismatch')
        base_h = finish_update(models['parent'], boundary, updates['base'])
        for cell, u in updates.items():
            label = branch+'/'+cell
            h = base_h if cell == 'base' else finish_update(models['parent'], boundary, u)
            scored = paired_scores(models['parent'], x, h, base_h)
            result[label] = {k: scored[k] for k in ('nll_sum', 'token_count')}
            if reference:
                result[label]['hidden'] = h.cpu()
                result[label]['update'] = u.cpu()
            # Predictor positions only, including the unscored causal prefix.
            d = (u-boundary['base'])[:, :-1].double()
            norms = u[:, :-1].double().norm(dim=-1)
            diagnostics[label] = dict(delta_energy=d.square().sum((1, 2)).cpu(),
                                       zero_update=(norms <= 1e-12).sum(1).cpu(),
                                       zero_delta=(d.norm(dim=-1) <= 1e-12).sum(1).cpu())
    return result, diagnostics


def summary(raw, diagnostic):
    ids = raw['row_ids']; counts = raw['token_count']
    if (set(raw) != set(LABELS)|{'row_ids', 'token_count'} or ids.ndim != 1 or
            ids.dtype.kind not in 'iu' or not len(ids) or len(np.unique(ids)) != len(ids) or
            counts.shape != ids.shape or counts.dtype.kind not in 'iu' or (counts <= 0).any()):
        raise ValueError('Invalid paired score arrays')
    expected = {f'{l}/{k}' for l in LABELS for k in ('delta_energy', 'zero_update', 'zero_delta')}
    if set(diagnostic) != expected:
        raise ValueError('Invalid diagnostic members')
    for v in list(raw.values())+list(diagnostic.values()):
        if v.shape != ids.shape or not np.isfinite(v).all() or (v < 0).any():
            raise ValueError('Invalid per-sequence values')
    for l in LABELS:
        for k in ('zero_update', 'zero_delta'):
            if diagnostic[l+'/'+k].dtype.kind not in 'iu':
                raise ValueError('Noninteger zero count')
    den = int(counts.sum()); cases = {}; gains = {}; energy = {}
    for l in LABELS:
        base = raw[l.split('/')[0]+'/base']; gain = (base-raw[l])/counts
        cases[l] = float(raw[l].sum()/den)
        gains[l] = dict(gain_nat=float((base-raw[l]).sum()/den),
                        q10_median_q90=np.quantile(gain, [.1, .5, .9]).tolist(),
                        positive_rows=int((gain > 0).sum()), negative_rows=int((gain < 0).sum()))
        energy[l] = {k: float(diagnostic[l+'/'+k].sum()) if k == 'delta_energy'
                     else int(diagnostic[l+'/'+k].sum()) for k in ('delta_energy', 'zero_update', 'zero_delta')}
    return dict(cases=cases, gains=gains, predictor_position_diagnostics=energy,
                sequences=len(ids), tokens=den, production_ready=False,
                scope='Frozen donor-dependent direction diagnostic; not standalone correction or causal shares',
                historical_document_disjointness_verified=False)


def evaluate_stream(models, c, proof, calibration_root, out):
    check_proof(c, proof); out = Path(out)
    if out.exists():
        raise ValueError('Fresh evaluation output required')
    rows, cal, ids = selections(c, models['parent'].c.length)
    calibration_root = Path(calibration_root); meta = read(calibration_root/'calibration.json')
    if meta['binding'] != binding(c, proof, cal, models['parent'].c.length):
        raise ValueError('Calibration/checkpoint binding mismatch')
    if c.get('calibration_artifact_sha256') != sha(calibration_root/'calibration.json'):
        raise ValueError('Calibration registration hash mismatch')
    if measurement_device() != c['measurement_device']:
        raise ValueError('Measurement runtime mismatch')
    fitted = replay_calibration(calibration_root)
    raw = {l: [] for l in LABELS}; diagnostic = {}; counts = []; out.mkdir(parents=True)
    for start in range(0, len(ids), c['batch_size']):
        x = rows.batch(ids[start:start+c['batch_size']], measurement_device())
        scored, d = panel(models, x, fitted); count = scored[LABELS[0]]['token_count'].numpy()
        for l in LABELS:
            if not np.array_equal(count, scored[l]['token_count'].numpy()):
                raise ValueError('Case denominators differ')
            raw[l].append(scored[l]['nll_sum'].numpy())
            for k, v in d[l].items():
                diagnostic.setdefault(l+'/'+k, []).append(v.numpy())
        counts.append(count)
        write_json(out/'progress.json', dict(rows=min(start+c['batch_size'], len(ids)), total=len(ids)))
    raw = {k: np.concatenate(v) for k, v in raw.items()}
    raw.update(row_ids=ids, token_count=np.concatenate(counts))
    diagnostic = {k: np.concatenate(v) for k, v in diagnostic.items()}
    report = summary(raw, diagnostic)
    np.savez_compressed(out/'scores.npz', **raw)
    np.savez_compressed(out/'diagnostics.npz', **diagnostic)
    write_json(out/'REPORT.json', report); write_json(out/'config.json', c)
    write_json(out/'calibration.json', meta)
    names = ('scores.npz', 'diagnostics.npz', 'REPORT.json', 'config.json', 'calibration.json')
    write_json(out/'manifest.json', dict(schema=1, labels=LABELS, input_pins=c['expected_inputs'],
        evaluation_row_ids=ids.tolist(), checkpoints=proof, measurement_runtime=measurement_policy(),
        files={n: sha(out/n) for n in names}, basis_exported=False, moments_exported=False))
    replay(out)
    return report


def replay(out):
    out = Path(out); m = read(out/'manifest.json')
    expected = {'scores.npz', 'diagnostics.npz', 'REPORT.json', 'config.json', 'calibration.json'}
    if m['labels'] != LABELS or set(m['files']) != expected:
        raise ValueError('Directional manifest registration mismatch')
    for name, h in m['files'].items():
        if sha(out/name) != h:
            raise ValueError('Directional output hash mismatch')
    c = read(out/'config.json'); validate(c); check_proof(c, m['checkpoints'])
    meta = read(out/'calibration.json'); b = meta['binding']
    if (c['expected_inputs'] != m['input_pins'] or
            c['measurement_device'] != m['measurement_runtime']['device'] or
            c['calibration_artifact_sha256'] != sha(out/'calibration.json') or
            b != binding(c, m['checkpoints'], np.array(b['calibration_row_ids']), b['length'])):
        raise ValueError('Directional configuration binding mismatch')
    with np.load(out/'scores.npz', allow_pickle=False) as f:
        raw = {k: f[k] for k in f.files}
    with np.load(out/'diagnostics.npz', allow_pickle=False) as f:
        diagnostic = {k: f[k] for k in f.files}
    if raw['row_ids'].tolist() != m['evaluation_row_ids'] or np.intersect1d(raw['row_ids'], b['calibration_row_ids']).size:
        raise ValueError('Replay row identity/overlap mismatch')
    if summary(raw, diagnostic) != read(out/'REPORT.json'):
        raise ValueError('Directional score replay mismatch')
    return dict(status='PASS', cases=len(LABELS), sequences=len(raw['row_ids']),
                model_reexecuted=False, node_basis_not_replayed_locally=True)
