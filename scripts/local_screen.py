"""Loop-screen arms on the local GPUs (4x A100) while the B200 is unavailable (26 Nov 2026).

Data: the DOT E2 training stream rebuilt locally and hash-verified (scripts/local_dot_e2_rebuild.py). The B200
row pins (train / measurement / calibration row ids) live on the B200, so this machine draws its own disjoint
rows from E2 once (`rows` action; numpy default_rng(20261126), single-document rows as Rows.eligible) and pins
them by SHA256. Every number from these runs is compared only with arms re-run here, never with B200 numbers.

The training / evaluation code is scripts/loop_screen_study.qualify and fit unchanged (one GPU per process).
  python scripts/local_screen.py rows --data /disk/thuat/dot_data_20261126/dot/e2/train --out <rows dir>
  python scripts/local_screen.py run --rows <rows dir> --base <run dir> --seed 0 --arm <arm>
  python scripts/local_screen.py queue --rows <rows dir> --base <run dir> --gpus 0,1,2,3 --jobs 0:armA 1:armB ...
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

TRAIN_ROWS = 24576 * 32  # enough for the 2x-token arms
MEASUREMENT_ROWS, CALIBRATION_ROWS = 512, 256


def sha(path):
    from efficient_llm.warm_order.data import sha as _sha
    return _sha(path)


def make_rows(data, out):
    from efficient_llm.warm_order.data import Rows
    out.mkdir(parents=True, exist_ok=False)
    rows = Rows(data)
    pool = rows.eligible()
    need = 2 * TRAIN_ROWS + MEASUREMENT_ROWS + CALIBRATION_ROWS
    if len(pool) < need:
        raise ValueError(f'Only {len(pool)} eligible rows, need {need}')
    pick = np.random.default_rng(20261126).choice(pool, need, replace=False)
    parts = dict(measurement=pick[:MEASUREMENT_ROWS], calibration=pick[MEASUREMENT_ROWS:MEASUREMENT_ROWS + CALIBRATION_ROWS],
                 train_s0=pick[MEASUREMENT_ROWS + CALIBRATION_ROWS:][:TRAIN_ROWS],
                 train_s1=pick[MEASUREMENT_ROWS + CALIBRATION_ROWS:][TRAIN_ROWS:])
    files = {}
    for name, ids in parts.items():
        np.save(out/f'{name}.npy', ids.astype(np.int64))
        files[name] = sha(out/f'{name}.npy')
    data_pins = {suffix: sha(str(data) + suffix) for suffix in ('.bin', '.starts')}
    (out/'manifest.json').write_text(json.dumps(dict(stem=str(data), data_sha256=data_pins, eligible=len(pool),
                                                     seed=20261126, files=files), indent=1) + '\n')
    print('LOCAL_ROWS', json.dumps(dict(eligible=len(pool), files=files)), flush=True)


def load(rows_dir):
    from efficient_llm.warm_order.data import Rows
    manifest = json.loads((rows_dir/'manifest.json').read_text())
    out = {}
    for name, digest in manifest['files'].items():
        if sha(rows_dir/f'{name}.npy') != digest:
            raise ValueError('Row pin changed: ' + name)
        out[name] = np.load(rows_dir/f'{name}.npy')
    return Rows(manifest['stem']), out, manifest


def run(rows_dir, base, seed, arm):
    from unittest import mock
    import scripts.loop_screen_study as loop
    from efficient_llm.warm_order.model import Config
    rows, ids, manifest = load(rows_dir)
    base.mkdir(parents=True, exist_ok=True)
    pins = (manifest['files']['train_s0'], manifest['files']['train_s1'])
    # a fit needing more rows than its own seed's array (4x-token arms) continues with the other seed's disjoint
    # array, as extended_ids does on the B200 (own rows first, then the other seed's)
    extended = lambda s: np.concatenate([ids[f'train_s{s}'], ids[f'train_s{1 - s}']])
    with mock.patch.object(loop, 'DEVICE', 'cuda'), mock.patch.object(loop, 'ROW_PINS', pins), \
            mock.patch.object(loop, 'extended_ids', extended):
        if not (base/f's{seed}'/f'qualification-{arm}'/'passed.json').is_file():
            loop.qualify(base, seed, arm, Config(), rows, ids['calibration'])
        loop.fit(base, seed, arm, {}, Config(), rows, ids[f'train_s{seed}'], ids['calibration'], ids['measurement'])


def queue(rows_dir, base, gpus, jobs):
    """One job per GPU at a time; a job whose summary exists is skipped; logs in <base>/logs."""
    (base/'logs').mkdir(parents=True, exist_ok=True)
    pending = [j for j in jobs if not (base/f's{j[0]}'/j[1]/'summary.json').is_file()]
    running = {}
    while pending or running:
        for gpu in gpus:
            if gpu not in running and pending:
                seed, arm = pending.pop(0)
                log = (base/'logs'/f's{seed}-{arm}.log').open('a')
                env = dict(__import__('os').environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
                           PYTHONDONTWRITEBYTECODE='1')
                cmd = [sys.executable, '-u', str(Path(__file__)), 'run', '--rows', str(rows_dir), '--base', str(base),
                       '--seed', str(seed), '--arm', arm]
                running[gpu] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT)), seed, arm)
                print(time.strftime('%H:%M:%S'), 'START', gpu, seed, arm, flush=True)
        for gpu, (proc, seed, arm) in list(running.items()):
            if proc.poll() is not None:
                print(time.strftime('%H:%M:%S'), 'END', gpu, seed, arm, 'rc', proc.returncode, flush=True)
                del running[gpu]
        time.sleep(10)
    print('QUEUE_DONE', flush=True)


def busy_gpus():
    """GPU indices with any compute process (read-only nvidia-smi)."""
    uuid = dict(line.split(', ')[::-1] for line in subprocess.run(
        ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], capture_output=True, text=True,
        check=True).stdout.strip().splitlines())
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'],
                          capture_output=True, text=True, check=True).stdout.split()
    return {int(uuid[u]) for u in apps if u in uuid}


CLAIMS = Path('/disk/thuat/dot_runs_local/gpu_claims')


def claimed_gpus():
    """GPUs claimed by a live job of any dynamic queue (claim file <gpu> holds the job PID), so a job that has not yet
    created its CUDA context is still seen as occupying its GPU."""
    import os
    out = set()
    for f in CLAIMS.glob('*') if CLAIMS.is_dir() else []:
        try:
            os.kill(int(f.read_text()), 0)
            out.add(int(f.name))
        except (ValueError, ProcessLookupError, PermissionError):
            f.unlink(missing_ok=True)
    return out


def queue_dynamic(rows_dir, base, gpus, jobs, after):
    """Like queue, but a job starts only on a GPU with no compute process, and only once every queue log in
    `after` (path:expected_starts) has started all its jobs, so no two queues ever share a GPU."""
    import os
    for item in after:
        path, need = item.rsplit(':', 1)
        while sum(' START ' in line for line in Path(path).read_text().splitlines()) < int(need):
            time.sleep(30)
    if after:  # let the last job started by another queue create its CUDA context before probing GPUs
        time.sleep(90)
    (base/'logs').mkdir(parents=True, exist_ok=True)
    pending = [j for j in jobs if not (base/f's{j[0]}'/j[1]/'summary.json').is_file()]
    running = {}
    while pending or running:
        for gpu, (proc, seed, arm) in list(running.items()):
            if proc.poll() is not None:
                print(time.strftime('%H:%M:%S'), 'END', gpu, seed, arm, 'rc', proc.returncode, flush=True)
                del running[gpu]
        busy = busy_gpus() | claimed_gpus()
        for gpu in gpus:
            if gpu not in running and gpu not in busy and pending:
                seed, arm = pending.pop(0)
                log = (base/'logs'/f's{seed}-{arm}.log').open('a')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', PYTHONDONTWRITEBYTECODE='1')
                cmd = [sys.executable, '-u', str(Path(__file__)), 'run', '--rows', str(rows_dir), '--base', str(base),
                       '--seed', str(seed), '--arm', arm]
                running[gpu] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT)), seed, arm)
                CLAIMS.mkdir(parents=True, exist_ok=True)
                (CLAIMS/str(gpu)).write_text(str(running[gpu][0].pid))
                print(time.strftime('%H:%M:%S'), 'START', gpu, seed, arm, flush=True)
        time.sleep(30)
    print('QUEUE_DONE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('rows', 'run', 'queue', 'queue_dynamic'))
    p.add_argument('--data', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--rows', type=Path)
    p.add_argument('--base', type=Path)
    p.add_argument('--seed', type=int)
    p.add_argument('--arm')
    p.add_argument('--gpus', default='0,1,2,3')
    p.add_argument('--jobs', nargs='*', default=[])
    p.add_argument('--after', nargs='*', default=[], help='queue_dynamic: <queue log>:<expected START count> ...')
    a = p.parse_args()
    if a.action == 'rows':
        make_rows(a.data, a.out)
    elif a.action == 'run':
        run(a.rows, a.base, a.seed, a.arm)
    else:
        args = (a.rows, a.base, [int(g) for g in a.gpus.split(',')],
                [(int(j.split(':')[0]), j.split(':', 1)[1]) for j in a.jobs])
        queue(*args) if a.action == 'queue' else queue_dynamic(*args, a.after)
