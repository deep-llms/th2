"""Bounded-batch descriptive geometry and node-side sufficient-statistic replay.

No fitting, remote submission, GPU admission, or document-freshness claim.
Large raw moments stay beside node results; only scalar reports are exportable.
"""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .adaptation_analysis import CANONICAL, SWAPPED, _frozen_cpu
from .adaptation_geometry import GeometryMoments, aligned_observations
from .data import Rows, sha
from .horizon_train import write_json


COMPARISONS = {
    'immediate': ('parent', CANONICAL, 'parent', SWAPPED),
    'ordinary_drift': ('parent', CANONICAL, 'A', CANONICAL),
    'swapped_drift': ('parent', SWAPPED, 'B', SWAPPED),
    'remaining': ('A', CANONICAL, 'B', SWAPPED),
}
KEYS = tuple(f'{alignment}/{i}/{field}' for alignment in ('slot', 'block') for i in range(6)
             for field in ('input', 'attention_update', 'ffn_update', 'output'))


def evaluate_geometry(models, comparison, data_stem, indices_path, expected_inputs,
                      batch_size, positions, keys, out, checkpoint_provenance=None,
                      max_moment_bytes=64*1024**2):
    if set(models) != {'parent', 'A', 'B'} or comparison not in COMPARISONS:
        raise ValueError('Registered parent/A/B comparison required')
    for model in models.values(): _frozen_cpu(model)
    config = asdict(models['parent'].c)
    if any(asdict(m.c) != config for m in models.values()): raise ValueError('Model configuration mismatch')
    if type(batch_size) is not int or batch_size <= 0: raise ValueError('Positive batch required')
    if (not isinstance(keys, tuple) or not keys or len(set(keys)) != len(keys) or
            any(key not in KEYS for key in keys)):
        raise ValueError('Explicit unique registered geometry keys required')
    if (not isinstance(positions, tuple) or not positions or len(set(positions)) != len(positions) or
            any(type(p) is not int or not 0 <= p < config['length'] for p in positions)):
        raise ValueError('Explicit valid token positions required')
    width = config['d']; moment_bytes = len(keys)*(3*width*width+2*width)*8
    if type(max_moment_bytes) is not int or max_moment_bytes <= 0 or moment_bytes > max_moment_bytes:
        raise ValueError('Moment buffer budget exceeded')
    out = Path(out)
    if out.exists() and any(out.iterdir()): raise ValueError('Fresh geometry output required')
    inputs = {p: sha(p) for p in (data_stem+'.bin', data_stem+'.starts', str(indices_path))}
    if inputs != expected_inputs: raise ValueError('Geometry input identity mismatch')
    ids = np.load(indices_path, allow_pickle=False)
    if ids.ndim != 1 or ids.dtype.kind not in 'iu' or not len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError('Unique nonempty geometry rows required')
    rows = Rows(data_stem, config['length'])
    if ids.min() < 0 or ids.max() >= len(rows.ids): raise ValueError('Geometry row out of range')
    left, path_left, right, path_right = COMPARISONS[comparison]
    moments = {key: GeometryMoments(width) for key in keys}; labels = None
    for start in range(0, len(ids), batch_size):
        samples = aligned_observations(models[left], models[right], rows.batch(ids[start:start+batch_size], 'cpu'),
                                       positions, path_x=path_left, path_y=path_right)
        current = {key: {k: v for k, v in samples[key].items() if k not in ('x', 'y')} for key in keys}
        if labels is None: labels = current
        elif current != labels: raise ValueError('Geometry alignment changed across batches')
        for key, moment in moments.items(): moment.update(samples[key]['x'], samples[key]['y'])
        del samples
    out.mkdir(parents=True, exist_ok=True)
    files = {}; reports = {}
    for i, key in enumerate(keys):
        name = f'moments_{i:03d}.npz'
        np.savez_compressed(out/name, **moments[key].arrays())
        files[name] = sha(out/name); reports[key] = moments[key].report()
    np.save(out/'row_ids.npy', ids, allow_pickle=False); files['row_ids.npy'] = sha(out/'row_ids.npy')
    write_json(out/'REPORT.json', reports); files['REPORT.json'] = sha(out/'REPORT.json')
    write_json(out/'manifest.json', dict(schema=1, comparison=comparison, keys=keys, labels=labels,
        positions=positions, config=config, files=files, provenance=checkpoint_provenance,
        verified_input_file_hashes=inputs, batch_size=batch_size, moment_buffer_bytes=moment_bytes,
        memory_scope='Moment tensors only; model/native traces/temporary copies/eigensolver excluded',
        observations_per_key=len(ids)*len(positions), raw_moments_exportable=False,
        production_ready=False, document_freshness_verified=False))
    replay_geometry(out)
    return reports


def replay_geometry(directory):
    directory = Path(directory); manifest = json.loads((directory/'manifest.json').read_text())
    keys = manifest['keys']; positions = manifest['positions']
    if (manifest['schema'] != 1 or not keys or len(set(keys)) != len(keys) or any(k not in KEYS for k in keys)
            or not positions or len(set(positions)) != len(positions)
            or any(type(p) is not int or not 0 <= p < manifest['config']['length'] for p in positions)):
        raise ValueError('Invalid geometry manifest')
    expected = {'REPORT.json', 'row_ids.npy'} | {f'moments_{i:03d}.npz' for i in range(len(keys))}
    if set(manifest['files']) != expected: raise ValueError('Unexpected geometry files')
    for name, digest in manifest['files'].items():
        if sha(directory/name) != digest: raise ValueError('Geometry file hash mismatch')
    ids = np.load(directory/'row_ids.npy', allow_pickle=False)
    if ids.ndim != 1 or ids.dtype.kind not in 'iu' or not len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError('Invalid geometry row IDs')
    n = len(ids)*len(positions)
    if manifest['observations_per_key'] != n: raise ValueError('Observation count mismatch')
    reports = {}
    for i, key in enumerate(keys):
        with np.load(directory/f'moments_{i:03d}.npz', allow_pickle=False) as raw:
            moment = GeometryMoments.from_arrays(dict(raw))
        if moment.n != n or moment.width != manifest['config']['d']:
            raise ValueError('Moment count/width mismatch')
        reports[key] = moment.report()
    if reports != json.loads((directory/'REPORT.json').read_text()):
        raise ValueError('Geometry numeric replay mismatch')
    return dict(status='PASS', keys=len(keys), observations_per_key=n, production_ready=False,
                scope='Node-side sufficient-statistic replay, not original-forward reexecution')
