"""DCLM paired-study row reservation against actual pinned stream bytes.

E2 boundary flags are row eligibility, NOT original-document identity. Known
historical exclusions remain incomplete. No model loading or remote action.
"""
import json
import copy
from pathlib import Path

import numpy as np

from .data import Rows, sha
from .horizon_train import write_json


def select(eligible, excluded, counts, seed):
    for a in (eligible, excluded):
        if a.ndim != 1 or a.dtype.kind not in 'iu' or len(np.unique(a)) != len(a) or (a < 0).any():
            raise ValueError('Unique nonnegative integer row IDs required')
    if (set(counts) != {'calibration', 'confirmation', 'train-s0', 'train-s1'} or
            any(type(n) is not int or n <= 0 for n in counts.values()) or type(seed) is not int or seed < 0):
        raise ValueError('Registered split sizes and seed required')
    pool = np.setdiff1d(eligible, excluded, assume_unique=True)
    needed = sum(counts.values())
    if needed > len(pool): raise ValueError('Insufficient eligible rows')
    order = np.random.default_rng(seed).choice(pool, needed, replace=False)
    groups = {}; offset = 0
    for name in ('calibration', 'confirmation', 'train-s0', 'train-s1'):
        n = counts[name]; groups[name] = order[offset:offset+n].astype(np.int64); offset += n
    return groups


def reserve(plan, out):
    """Require actual stream bytes; never emit READY from metadata or fixtures."""
    out = Path(out)
    if out.exists(): raise ValueError('Fresh reservation output required')
    if plan['schema'] != 1 or plan['domain'] != 'dclm' or plan['row_space'] != 'E2_destination':
        raise ValueError('Registered DCLM E2 row space required')
    for key in ('length', 'stream_rows', 'steps', 'batch', 'calibration_rows', 'confirmation_rows'):
        if type(plan[key]) is not int or plan[key] <= 0: raise ValueError('Positive count required: '+key)
    inputs = {p: sha(p) for p in (plan['data']+'.bin', plan['data']+'.starts')}
    if inputs != plan['expected_stream_hashes']: raise ValueError('Actual stream hash mismatch')
    if sha(plan['exposure_proof']) != plan['exposure_proof_sha256']:
        raise ValueError('Exposure proof hash mismatch')
    proof = json.loads(Path(plan['exposure_proof']).read_text())
    if (proof['token_sha256'] != inputs[plan['data']+'.bin'] or
            proof['starts_sha256'] != inputs[plan['data']+'.starts'] or
            sha(plan['exposure_rows']) != proof['output_sha256']):
        raise ValueError('Exposure inventory stream identity mismatch')
    with np.load(plan['exposure_rows'], allow_pickle=False) as raw:
        excluded = raw['known_e2_rows'].copy()
    if len(excluded) != proof['known_union_rows']: raise ValueError('Known exclusion count mismatch')
    rows = Rows(plan['data'], plan['length'])
    if len(rows.ids) != plan['stream_rows'] or (excluded >= len(rows.ids)).any():
        raise ValueError('Stream row extent mismatch')
    eligible = rows.eligible()
    counts = dict(calibration=plan['calibration_rows'], confirmation=plan['confirmation_rows'],
                  **{'train-s0':plan['steps']*plan['batch'], 'train-s1':plan['steps']*plan['batch']})
    groups = select(eligible, excluded, counts, plan['selection_seed'])
    out.mkdir(parents=True)
    write_json(out/'plan.json', plan)
    files = {'plan.json':sha(out/'plan.json')}
    for name, values in groups.items():
        np.save(out/(name+'.npy'), values, allow_pickle=False)
        files[name+'.npy'] = sha(out/(name+'.npy'))
    report = dict(schema=1, status='ROW_BYTES_VERIFIED_ONLY', files=files, counts=counts,
        actual_stream_hashes=inputs, eligible_rows=len(eligible), excluded_known_rows=len(excluded),
        new_split_row_overlap=0, known_inventory_row_overlap=0,
        row_space=plan['row_space'], paired_AB_same_rows=True, seed_training_rows_disjoint=True,
        numpy_version=np.__version__, selection_seed=plan['selection_seed'],
        historical_training_overlap='unknown beyond authenticated partial inventory',
        whole_document_disjointness_verified=False, production_ready=False,
        flags=['exploratory', 'potential_historical_reuse', 'no_original_document_freshness_claim'])
    write_json(out/'manifest.json', report)
    verify(out)
    return report


def verify(out):
    """Verify saved arrays against each other and the pinned partial exclusions."""
    out = Path(out); m = json.loads((out/'manifest.json').read_text())
    expected = {'plan.json', 'calibration.npy', 'confirmation.npy', 'train-s0.npy', 'train-s1.npy'}
    if m['schema'] != 1 or set(m['files']) != expected: raise ValueError('Reservation file set mismatch')
    for name, digest in m['files'].items():
        if sha(out/name) != digest: raise ValueError('Reservation hash mismatch')
    plan = json.loads((out/'plan.json').read_text())
    if sha(plan['exposure_proof']) != plan['exposure_proof_sha256']: raise ValueError('Exposure proof mismatch')
    proof = json.loads(Path(plan['exposure_proof']).read_text())
    if sha(plan['exposure_rows']) != proof['output_sha256']: raise ValueError('Exposure rows mismatch')
    stream = plan['expected_stream_hashes']
    if (m['actual_stream_hashes'] != stream or proof['token_sha256'] != stream[plan['data']+'.bin'] or
            proof['starts_sha256'] != stream[plan['data']+'.starts']):
        raise ValueError('Registered stream identity mismatch')
    groups = {name:np.load(out/(name+'.npy'), allow_pickle=False) for name in ('calibration','confirmation','train-s0','train-s1')}
    required = dict(calibration=plan['calibration_rows'], confirmation=plan['confirmation_rows'],
                    **{'train-s0':plan['steps']*plan['batch'], 'train-s1':plan['steps']*plan['batch']})
    if m['counts'] != required: raise ValueError('Registered counts mismatch')
    for name, ids in groups.items():
        if (ids.ndim != 1 or ids.dtype != np.int64 or len(ids) != required[name] or
                (ids < 0).any() or (ids >= plan['stream_rows']).any()):
            raise ValueError('Invalid reserved row array')
    all_ids = np.concatenate(list(groups.values()))
    if len(np.unique(all_ids)) != len(all_ids): raise ValueError('Reservation row overlap')
    with np.load(plan['exposure_rows'], allow_pickle=False) as raw:
        if np.intersect1d(all_ids, raw['known_e2_rows']).size: raise ValueError('Known exposure overlap')
    return dict(status='PASS', row_counts=required, production_ready=False,
                scope='Array/hash/disjointness replay; admission must reverify source bytes and boundaries')


def bind(config, directory, manifest_sha256, seed):
    """Explicitly replace row paths with a pinned reservation; retain train policy."""
    if type(seed) is not int or seed not in (0, 1): raise ValueError('Parent seed0/1 required')
    directory = Path(directory)
    if sha(directory/'manifest.json') != manifest_sha256: raise ValueError('Reservation manifest hash mismatch')
    verify(directory)
    plan = json.loads((directory/'plan.json').read_text())
    for key in ('data', 'steps', 'batch', 'domain'):
        if config[key] != plan[key]: raise ValueError('Training/reservation policy mismatch: '+key)
    c = copy.deepcopy(config)
    c.update(indices=str(directory/f'train-s{seed}.npy'),
             calibration_indices=str(directory/'calibration.npy'),
             confirmation_indices=str(directory/'confirmation.npy'),
             row_reservation=dict(directory=str(directory), manifest_sha256=manifest_sha256, seed=seed))
    m = json.loads((directory/'manifest.json').read_text())
    c['expected_inputs'] = dict(plan['expected_stream_hashes'])
    for key in ('indices', 'calibration_indices', 'confirmation_indices'):
        c['expected_inputs'][c[key]] = m['files'][Path(c[key]).name]
    return c


def validate_binding(config):
    """Check recipe binding before loading/updating a model; no source-data scan."""
    spec = config['row_reservation']
    if set(spec) != {'directory', 'manifest_sha256', 'seed'}:
        raise ValueError('Invalid reservation binding schema')
    expected = bind(config, spec['directory'], spec['manifest_sha256'], spec['seed'])
    if config != expected: raise ValueError('Training row reservation binding mismatch')
    return json.loads((Path(spec['directory'])/'plan.json').read_text())
