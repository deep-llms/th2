"""Hash-pinned CPU snapshot loader and bounded-batch evaluation integration.

No production admission, GPU execution, remote operations or fabricated row
reservation. In particular, caller row pins do not prove historical freshness.
"""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json

import numpy as np
import torch

from .adaptation_analysis import CANONICAL, SWAPPED
from .adaptation_train import load_parent, validate_adam, source_hashes
from .adaptation_eval import collect, write_results, replay
from .data import Rows, sha
from .model import Config, Model


def load_snapshot_panel(parent_directory, parent_pins, phase_files):
    """phase_files: {A/B: {path, sha256}} for the same continuation update.

    Pins must come from an authenticated caller manifest. The loader verifies
    bytes and their semantic pairing, not the authority of the manifest itself.
    """
    if set(phase_files) != {'A', 'B'}:
        raise ValueError('Both phase checkpoint pins required')
    with torch.random.fork_rng(devices=[]):
        parent_model, optimizer, parent, names = load_parent(parent_directory, parent_pins)
        lr = optimizer.param_groups[0]['lr']
        del optimizer
        models, states = {'parent': parent_model.eval().requires_grad_(False)}, {}
        for branch, spec in phase_files.items():
            if set(spec) != {'path', 'sha256'} or sha(spec['path']) != spec['sha256']:
                raise ValueError('Phase checkpoint hash mismatch')
            state = torch.load(spec['path'], map_location='cpu', weights_only=False)
            recipe = state['recipe']; c = recipe['config']; step = state['step']
            if (c['branch'] != branch or recipe['parent_recipe'] != parent['recipe'] or
                    c['parent_hashes'] != parent_pins or recipe['parameter_names'] != names or
                    recipe['source'] != source_hashes() or recipe['precision'] != 'fp32'):
                raise ValueError('Phase source/parent/branch/name identity mismatch')
            if recipe['paired_common'] != {k: v for k, v in c.items() if k != 'branch'}:
                raise ValueError('Phase common recipe inconsistent')
            if type(step) is not int or not 0 <= step <= c['steps']:
                raise ValueError('Invalid phase update')
            path = CANONICAL if branch=='A' else SWAPPED
            counts = {','.join(map(str, path)): step} if step else {}
            if (state['counts'] != counts or
                    state['block_tokens'] != step*c['batch']*parent_model.c.length*6):
                raise ValueError('Phase path accounting mismatch')
            model = Model(Config(**parent['recipe']['config']['model']))
            expected = model.state_dict()
            if expected.keys() != state['model'].keys() or any(
                    value.shape != expected[key].shape or value.dtype != expected[key].dtype or
                    not torch.isfinite(value).all() for key, value in state['model'].items()):
                raise ValueError('Phase tensor schema/nonfinite mismatch')
            validate_adam(model, state['optimizer'], parent['step']+step, lr)
            model.load_state_dict(state['model'], strict=True)
            models[branch] = model.eval().requires_grad_(False)
            states[branch] = dict(step=step, common=recipe['paired_common'])
        if states['A'] != states['B']:
            raise ValueError('Unmatched continuation steps or recipes')
    return models, dict(parent_file_pins=parent_pins, phase_files=phase_files,
                        paired_recipe_sha256=hashlib.sha256(json.dumps(states['A']['common'], sort_keys=True).encode()).hexdigest(),
                        phase_step=states['A']['step'], checkpoint_bytes_and_pairing_verified=True,
                        authority_of_external_pins='caller responsibility', production_ready=False)


def evaluate_stream(models, data_stem, indices_path, expected_inputs, batch_size, out,
                    checkpoint_provenance=None):
    """Keep activations bounded by batch; retain only O(rows × cases) scores.

    All three exact input files are hashed before model evaluation. This confirms
    selected bytes, not document freshness or independence. CPU only via collect.
    """
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError('Positive batch size required')
    out = Path(out)
    if out.exists() and any(out.iterdir()): raise ValueError('Fresh output required')
    inputs = {p: sha(p) for p in (data_stem+'.bin', data_stem+'.starts', str(indices_path))}
    if inputs != expected_inputs: raise ValueError('Evaluation input identity mismatch')
    ids = np.load(indices_path, allow_pickle=False)
    if ids.ndim != 1 or ids.dtype.kind not in 'iu' or not len(ids) or len(np.unique(ids)) != len(ids):
        raise ValueError('Unique nonempty evaluation rows required')
    rows = Rows(data_stem, models['parent'].c.length)
    if ids.min() < 0 or ids.max() >= len(rows.ids): raise ValueError('Evaluation row out of range')
    pieces, labels, maxima = {}, None, {}
    for start in range(0, len(ids), batch_size):
        ix = ids[start:start+batch_size]
        raw, current, parity = collect(models, rows.batch(ix, 'cpu'), ix)
        if labels is None: labels = current
        elif labels != current: raise ValueError('Case order changed across batches')
        for key, values in raw.items(): pieces.setdefault(key, []).append(values)
        for family, values in parity.items():
            dest = maxima.setdefault(family, {})
            for key, value in values.items(): dest[key] = max(dest.get(key, 0.), value)
    raw = {key: np.concatenate(values) for key, values in pieces.items()}
    if not np.array_equal(raw['row_ids'], ids): raise ValueError('Evaluation row order changed')
    provenance = dict(checkpoints=checkpoint_provenance, verified_input_file_hashes=inputs,
                      batch_size=batch_size, whole_document_freshness_verified=False,
                      production_ready=False)
    result = write_results(raw, labels, maxima, asdict(models['parent'].c), out, provenance)
    replay(out)
    return result
