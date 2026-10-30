"""Atomic progress snapshots inside a fresh, experiment-owned quantization root.

The generic eval writer is create-only; these tests and decisions evolve as
registered stages finish. Preserve immutable inputs through SHA locks, and
replace only this experiment's progress/output snapshots.
"""
import json
from pathlib import Path
from eval.a2_run import lock


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    encoded=json.dumps(value,indent=2,allow_nan=False)+'\n'
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(encoded)
    temporary.replace(path)
    return lock(path)
