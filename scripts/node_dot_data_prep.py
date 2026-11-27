"""Prepare the DOT E2 data, row pins and downstream requests on a fresh GPU node (28 Nov 2026, B200 pq5r), CPU only.

Inputs are the controller `#d` downloads under --root (tokenizer/, raw/dclm/<index>.parquet for the first 20 pinned
DCLM shards of resources/a2_corpora_30b_20261029.json, downstream/<manifest layout>). A `#d` dataset destination can
be a directory holding the file (observed on OHMV, 18 Nov 2026); each expected path is resolved to the single file it
names or contains, and symlinked into <root>/norm so the unchanged local rebuild sees plain files.
Stages (each recorded in <out>/status.json; <out>/complete.json only after every requested stage passed):
  resolve     locate every expected file (20 DCLM shards, 3 tokenizer files)
  verify      SHA256 and size of every DCLM shard / tokenizer file against the pinned recipe
  e2          scripts/local_dot_e2_rebuild.py on <root>/norm (must report MATCH against dot_e2_expected)
  rows        scripts/local_screen.py rows from the rebuilt E2; every row-array SHA must equal the local pins
              (docs/results/local_rows_20261126/manifest.json), so node and local A100 runs share their rows
  downstream  eval.a2_requests prepare (eval env) on the pinned downstream files; requests.json SHA must equal
              the pinned 6282100c... (scripts/local_downstream.REQUESTS_SHA)
  python scripts/node_dot_data_prep.py --root /mnt/local/_data/deep-llms_th2/dot_data_20261128 --out <status dir>
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT/'resources/a2_corpora_30b_20261029.json'
ROW_PINS = ROOT/'docs/results/local_rows_20261126/manifest.json'
REQUESTS_SHA = '6282100c39545c6d14b219a4a6538e4ecf13b94a12b97c19c68c18c61a450091'
SHARDS = 20


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def resolve(path):
    """The file a #d destination names: the path itself, or the single regular file inside that directory."""
    path = Path(path)
    if path.is_file():
        return path
    if path.is_dir():
        files = [p for p in path.iterdir() if p.is_file()]
        if len(files) == 1:
            return files[0]
        raise ValueError(f'{path}: expected one file in the download directory, found {len(files)}')
    raise FileNotFoundError(path)


class Status:
    def __init__(self, out):
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        self.state = dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), stages={})

    def set(self, stage, **kw):
        self.state['stages'][stage] = dict(kw, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        tmp = self.out/'status.json.tmp'
        tmp.write_text(json.dumps(self.state, indent=1) + '\n')
        os.replace(tmp, self.out/'status.json')
        print('STAGE', stage, json.dumps(kw), flush=True)


def run(cmd, log):
    with open(log, 'a') as f:
        f.write('$ ' + ' '.join(map(str, cmd)) + '\n')
        f.flush()
        return subprocess.run(list(map(str, cmd)), cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT).returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--stages', nargs='+', default=['resolve', 'verify', 'e2', 'rows', 'downstream'])
    p.add_argument('--workers', type=int, default=20)
    p.add_argument('--python', default=sys.executable, help='training env interpreter (E2 rebuild, rows)')
    p.add_argument('--eval-python', default='/mnt/local/conda-py311/envs/eval/bin/python3.11')
    a = p.parse_args()
    status = Status(a.out)
    recipe = json.loads(RECIPE.read_text())
    shards = recipe['corpora']['dclm']['files'][:SHARDS]
    norm = a.root/'norm'
    expected = {f"raw/dclm/{s['index']:05d}.parquet": (s['sha256'], s['bytes']) for s in shards}
    expected.update({f'tokenizer/{name}': (digest, None) for name, digest in recipe['tokenizer_files'].items()})
    try:
        if 'resolve' in a.stages:
            found = {}
            for rel in expected:
                src = resolve(a.root/rel)
                dst = norm/rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.is_symlink() and dst.resolve() != src.resolve():
                    raise ValueError(f'{dst} already points elsewhere')
                if not dst.exists():
                    dst.symlink_to(src)
                found[rel] = str(src.relative_to(a.root))
            status.set('resolve', ok=True, files=len(found), layout=found)
        if 'verify' in a.stages:
            bad = []
            for rel, (digest, size) in expected.items():
                path = (norm/rel).resolve()
                if (size is not None and path.stat().st_size != size) or sha(path) != digest:
                    bad.append(rel)
            if bad:
                raise ValueError(f'hash/size mismatch: {bad}')
            status.set('verify', ok=True, files=len(expected))
        if 'e2' in a.stages:
            rc = run([a.python, '-u', 'scripts/local_dot_e2_rebuild.py', '--root', norm, '--shards', SHARDS,
                      '--workers', a.workers], a.out/'e2.log')
            proof = json.loads((norm/'dot/e2_local_rebuild.json').read_text()) if (norm/'dot/e2_local_rebuild.json').is_file() else {}
            if rc or not proof.get('ok'):
                raise ValueError(f'E2 rebuild failed (rc {rc}, ok {proof.get("ok")}); see e2.log')
            status.set('e2', ok=True, checks={k: v[0] for k, v in proof['checks'].items()})
        if 'rows' in a.stages:
            rows = norm/'local_rows'
            rc = run([a.python, '-u', 'scripts/local_screen.py', 'rows', '--data', norm/'dot/e2/train', '--out', rows],
                     a.out/'rows.log')
            got = json.loads((rows/'manifest.json').read_text())['files'] if (rows/'manifest.json').is_file() else {}
            want = json.loads(ROW_PINS.read_text())['files']
            if rc or got != want:
                raise ValueError(f'row pins differ from the local pins (rc {rc})')
            status.set('rows', ok=True, files=got)
        if 'downstream' in a.stages:
            out = a.root/'downstream'/'requests.json'
            if not out.is_file():
                rc = run([a.eval_python, '-m', 'eval.a2_requests', 'prepare', '--tokenizer', norm/'tokenizer',
                          '--dataset-root', a.root/'downstream', '--manifest', 'resources/warm_downstream_data_20261118.json',
                          '--tasks', 'hellaswag', 'piqa', 'arc_easy', 'lambada_openai', '--max-length', 512,
                          '--output', out], a.out/'downstream.log')
                if rc:
                    raise ValueError(f'downstream prepare failed (rc {rc}); see downstream.log')
            digest = sha(out)
            if digest != REQUESTS_SHA:
                raise ValueError(f'requests.json SHA {digest} differs from the pinned {REQUESTS_SHA}')
            status.set('downstream', ok=True, requests_sha256=digest)
    except Exception as e:
        status.set('failed', ok=False, error=f'{type(e).__name__}: {e}')
        raise
    (a.out/'complete.json').write_text(json.dumps(dict(ok=True, stages=a.stages, root=str(a.root)), indent=1) + '\n')
    print('NODE_DATA_PREP_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
