"""Join replayed, matched snapshot panels; no model execution or fitted probe."""
import json
from pathlib import Path

import numpy as np

from .adaptation_eval import replay
from .data import sha


def summarize(directories):
    """Require step zero and identical parent, recipe, data and score membership.

    Inputs are existing authenticated-by-caller panel directories. File hashes
    and numeric reports are replayed, not treated as independent authenticity.
    No extrapolation, recovery percentage, or automatic mechanism verdict.
    """
    if len(directories) < 2: raise ValueError('Baseline and later panel required')
    points, sources, signature, row_ids, counts = [], [], None, None, None
    for directory in map(Path, directories):
        replay(directory)
        manifest = json.loads((directory/'manifest.json').read_text())
        report = json.loads((directory/'REPORT.json').read_text())
        provenance = manifest.get('provenance') or {}
        ckpt = provenance.get('checkpoints') or {}
        if not ckpt.get('checkpoint_bytes_and_pairing_verified'):
            raise ValueError('Verified paired checkpoint provenance required')
        current = dict(parent=ckpt['parent_file_pins'], recipe=ckpt['paired_recipe_sha256'],
                       inputs=provenance['verified_input_file_hashes'],
                       config=manifest['config'], cases=manifest['cases'], batch=provenance['batch_size'],
                       measurement_runtime=manifest.get('measurement_runtime'))
        if signature is None: signature = current
        elif current != signature: raise ValueError('Unmatched parent/recipe/data/config/cases/batch')
        step = ckpt['phase_step']
        if type(step) is not int or step < 0 or (points and step <= points[-1]['step']):
            raise ValueError('Unique increasing snapshot updates required')
        if not points and step != 0: raise ValueError('Step-zero baseline required')
        with np.load(directory/'scores.npz', allow_pickle=False) as raw:
            if row_ids is None:
                row_ids, counts = raw['row_ids'].copy(), raw['0/token_count'].copy()
            elif not np.array_equal(row_ids, raw['row_ids']) or not np.array_equal(counts, raw['0/token_count']):
                raise ValueError('Unpaired observations/denominators across snapshots')
        losses = {r['case']: r['nll'] for r in report['cases']}
        points.append(dict(step=step, nll=losses, contrasts_nat_per_token=report['contrasts_nat_per_token']))
        sources.append(dict(directory=str(directory), manifest_sha256=sha(directory/'manifest.json'),
                            files=manifest['files']))
    baseline = points[0]['nll']
    # Step zero must be the exact fork, not an arbitrary first trained checkpoint.
    for branch in ('A', 'B'):
        for order in ('canonical', 'swap'):
            if abs(baseline[branch+'/'+order]-baseline['parent/'+order]) > 2e-6:
                raise ValueError('Step-zero parent/fork score mismatch')
    for point in points:
        loss = point['nll']
        if any(abs(loss['parent/'+p]-baseline['parent/'+p]) > 2e-6 for p in ('canonical', 'swap')):
            raise ValueError('Parent reference changed across snapshots')
        a = baseline['A/canonical']-loss['A/canonical']
        b = baseline['B/swap']-loss['B/swap']
        point['ordinary_gain_nat'] = a
        point['swapped_gain_nat'] = b
        point['excess_gain_over_ordinary_nat'] = b-a
        point['remaining_gap_nat'] = loss['B/swap']-loss['A/canonical']
    return dict(points=points, sources=sources, pairing=signature,
                sequences=len(row_ids), production_ready=False,
                interpretation='Paired finite-sample trajectory; excess recovery is not a mechanism attribution',
                historical_document_disjointness_verified=False)
