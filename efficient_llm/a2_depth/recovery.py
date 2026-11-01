"""Durable C recovery evidence; no GPU allocation or numerical substitutions."""
from pathlib import Path
from contextlib import contextmanager
import fcntl
from .common import Checks, read, write
from scripts.a2_quant_io import lock
from scripts.a2_remaining import read_locked


def continued_checks(path):
    checks=Checks(path)
    if Path(path).exists():checks.items=read(path)['checks']
    return checks


@contextmanager
def coordinator_lock(root):
    with (Path(root)/'coordinator.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def seal_arm(out, identity):
    out=Path(out)
    files=[out/'result.json',out/'recipe.json',out/'order.json',*sorted(out.glob('tests-rank*.json')),
           *sorted(p for p in (out/'final').rglob('*') if p.is_file())]
    return write(out/'arm-complete.json',dict(identity=identity,files=[lock(p) for p in files]))


def completed_arm(out, identity):
    out=Path(out);marker=read(out/'arm-complete.json')
    if marker['identity']!=identity:raise ValueError('Completed arm recipe/source changed')
    paths={Path(item['path']).resolve() for item in marker['files']}
    required={out/'result.json',out/'recipe.json',out/'order.json',
              *[out/f'tests-rank{rank}.json' for rank in range(8)],*sorted(p for p in (out/'final').rglob('*') if p.is_file())}
    if paths!={p.resolve() for p in required} or not any(p.suffix=='.safetensors' for p in paths):
        raise ValueError('Completed arm member coverage changed')
    for item in marker['files']:read_bytes_locked(item)
    result=read(out/'result.json')
    if result['status']!='complete' or result['full_state_identity']!=identity or len(result['lr_updates'])!=identity['steps']:
        raise ValueError('Incomplete registered arm')
    return marker


def read_bytes_locked(item):
    from scripts.train_then_burn import digest
    if digest(item['path'])!=item['sha256']:raise ValueError('Changed recovery artifact: '+item['path'])


def next_path(path):
    path=Path(path)
    if not path.exists():return path
    i=1
    while True:
        candidate=path.with_name(path.stem+f'-attempt{i:03d}'+path.suffix)
        if not candidate.exists():return candidate
        i+=1
