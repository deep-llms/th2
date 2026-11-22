"""Bind the pilot's prefix to an authenticated existing DCLM reservation.

Whole-file SHA pins precede read-only memmaps. Does not establish original
document freshness or certify that source files remain immutable after loading.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .adaptation_reservation import validate_binding
from .data import Rows, sha


class BoundRows:
    def __init__(self, recipe_pin, *, seed, length=512, fixture_only=False):
        if type(fixture_only) is not bool or seed not in (0, 1):
            raise ValueError('Explicit fixture flag and seed0/1 required')
        if sha(recipe_pin['path']) != recipe_pin['sha256']:
            raise ValueError('Prior recipe hash mismatch')
        previous = json.loads(Path(recipe_pin['path']).read_text())['config']
        # Production continuation recipes bind seed inside row_reservation, not at top level.
        bound_seed = previous.get('row_reservation', {}).get('seed', previous.get('seed'))
        if (previous['domain'] != 'dclm' or bound_seed != seed
                or ('seed' in previous and previous['seed'] != seed)
                or previous['steps'] * previous['batch'] != 64000):
            raise ValueError('Expected existing DCLM64000-row recipe for this seed')
        if not fixture_only:
            validate_binding(previous)
            if previous['row_reservation']['seed'] != seed or length != 512:
                raise ValueError('Reservation seed/length mismatch')
        paths = [previous['data'] + ext for ext in ('.bin', '.starts')]
        paths += [previous[k] for k in ('indices', 'calibration_indices', 'confirmation_indices')]
        pins = {p: sha(p) for p in paths}
        if any(previous['expected_inputs'].get(p) != v for p, v in pins.items()):
            raise ValueError('Actual corpus/row-array hash mismatch')
        self.rows = Rows(previous['data'], length)
        arrays = [np.load(previous[k], allow_pickle=False)
                  for k in ('indices', 'calibration_indices', 'confirmation_indices')]
        for array, count in zip(arrays, (64000, 256, 512)):
            if (array.shape != (count,) or array.dtype.kind not in 'iu'
                    or len(np.unique(array)) != count or array.min() < 0
                    or array.max() >= len(self.rows.ids)):
                raise ValueError('Invalid selected rows/extent')
        for i in range(3):
            for j in range(i):
                if np.intersect1d(arrays[i], arrays[j]).size:
                    raise ValueError('Train/calibration/measurement row overlap')
        for array in arrays:
            for start in range(0, len(array), 4096):
                if self.rows.starts[array[start:start + 4096], 1:].any():
                    raise ValueError('Selected row crosses a packed boundary')
        self.train_ids, self.calibration_ids, self.measurement_ids = (
            arrays[0][:61440].astype(np.int64), arrays[1].astype(np.int64), arrays[2].astype(np.int64))
        self.length = length
        self.binding = dict(schema=1, prior_recipe_pin=dict(recipe_pin), seed=seed,
                            fixture_only=fixture_only, input_pins=pins, length=length,
                            train_prefix_rows=61440, calibration_rows=256, measurement_rows=512,
                            train_prefix_sha256=hashlib.sha256(self.train_ids.astype('<i8').tobytes()).hexdigest(),
                            document_freshness_verified=False, historical_exposure_unknown=True)

    def batch(self, ids):
        return self.rows.batch(ids, 'cpu')

    def training_batch(self, step):
        if type(step) is not int or not 0 <= step < 1920:
            raise ValueError('Invalid lineage step')
        ids = self.train_ids[step * 32:(step + 1) * 32]
        return ids.copy(), self.batch(ids)
