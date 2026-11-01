"""Saved-array coverage and inherited scientific flags, separate from execution."""
import numpy as np
from .common import read, SPEC_SHA


def collect_checks(root):
    checks = []
    for name in ('preparation_checks.json', 'coordinator_checks.json'):
        path = root/name
        if path.exists():
            try:
                checks.extend(dict(item, check_source=name) for item in read(path)['checks'])
            except Exception as error:
                checks.append(dict(test='read_checks:'+name, passed=False, detail=repr(error)))
    return checks


def array_audit(row, cfg, name):
    """2461 x 4095 is the locked A2 final set; fixtures explicitly override it.

    Finite but incomplete arrays can still be described. They never establish
    complete scientific coverage, even when all expected filenames exist.
    """
    arr = np.asarray(row['per_sequence_nll'], dtype=np.float64)
    expected = cfg.get('final_sequences', 2461)
    width = cfg.get('final_targets_per_sequence', 4095)
    usable = arr.ndim == 1 and len(arr) > 0 and np.isfinite(arr).all()
    conditions = dict(
        finite_vector=bool(usable),
        full_sequence_count=arr.ndim == 1 and len(arr) == expected,
        declared_sequence_count=row.get('sequences') == expected,
        token_count=row.get('tokens') == expected*width,
        spec_identity=row.get('spec_sha256') == SPEC_SHA,
        saved_mean=bool(usable and np.isfinite(row.get('loss', float('nan')))
                        and abs(float(arr.mean())-row['loss']) <= 1e-12))
    checks = [dict(test=key+':'+name, passed=bool(ok), spec_sha256=SPEC_SHA)
              for key, ok in conditions.items()]
    for flag in row.get('flags', []):
        checks.append(dict(test='inherited:'+name, passed=False, detail=flag, spec_sha256=SPEC_SHA))
    if row.get('unverified') and not row.get('flags'):
        checks.append(dict(test='inherited_unverified:'+name, passed=False, spec_sha256=SPEC_SHA))
    return arr, bool(usable), all(conditions.values()), checks
