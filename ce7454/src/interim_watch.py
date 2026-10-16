"""Bounded CPU controller; one small GPU0 evaluator alongside authorized training.

Never signals trainers. Stops/reaps only its own evaluator before final-eval gate
releases the original burn supervisor. All model checkpoints are read-only.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from attach_band_study import identity, original_running, save, stop_owned

BASE = Path('/mnt/local/_outputs/deep-llms_th2')
ORIGINAL = BASE/'q359-ce7454-scratch-20261016-a04'
BAND = BASE/'q359-ce7454-band-20261016-a01'
CACHE = '/mnt/local/_data/deep-llms_th2/ce7454/evaluation/cache'
# Early priority; all eight are evaluated, never selected by test labels.
ARMS = [('grip_nodown', 'grip_nodown', ORIGINAL/'supervised/run/study'),
        ('band_strong', 'band_stem', BAND/'study'),
        ('band_ibn_strong', 'band_stem_ibn', BAND/'study'),
        ('srresnet', 'srresnet', ORIGINAL/'supervised/run/study'),
        ('band_moderate', 'band_stem', BAND/'study'),
        ('phase_strong', 'phase_stem', BAND/'study'),
        ('grip_nodown_stats', 'grip_nodown_stats', ORIGINAL/'supervised/run/study'),
        ('srresnet_stats', 'srresnet_stats', ORIGINAL/'supervised/run/study')]


def checkpoint_for(name, root):
    if root == BAND/'study':
        return root/name/'last.pt'
    final = root/(name+'_full512')/'last.pt'
    return final if final.exists() else root/(name+'_base')/'last.pt'


def should_stop(output, owner, end):
    return ((output/'STOP').exists() or time.time() >= end
            or not original_running(ORIGINAL, owner))


def observe(output, cycle):
    gpu = subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                          '--format=csv'], capture_output=True, text=True, check=True).stdout
    progress = {}
    for name, _, root in ARMS:
        progress[name] = {}
        for path in root.glob(name+'*/log.csv'):
            lines = path.read_text().splitlines()
            progress[name][str(path)] = lines[-2:]
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'cycle': cycle,
              'gpu': gpu, 'progress': progress}
    save(output/'latest_status.json', result)
    print('WATCH_STATUS '+json.dumps(result), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--once', action='store_true')
    p.add_argument('--arms', nargs='+', default=[x[0] for x in ARMS])
    p.add_argument('--interval', type=int, default=1800)
    p.add_argument('--finish-by', default='2026-10-17T02:00:00+00:00')
    args = p.parse_args()
    if args.interval < 60 or not set(args.arms) <= {x[0] for x in ARMS}:
        raise ValueError('Invalid watcher settings')
    if args.output.exists():
        raise FileExistsError(args.output)
    lease = json.loads(Path('/mnt/local/_gpu_guard/DISABLED').read_text())
    if lease['output'] != str(ORIGINAL/'supervised'):
        raise ValueError('Another GPU owner')
    owner = identity(lease['owner_pid'])
    if not original_running(ORIGINAL, owner):
        raise ValueError('Original training no longer active')
    args.output.mkdir(parents=True)
    end = datetime.fromisoformat(args.finish_by).timestamp()
    save(args.output/'owner.json', {'owner': owner, 'watcher': identity(os.getpid())})
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Watcher signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    child = None
    cycle = 0
    status = 'passed'
    seen = {}
    try:
        while not should_stop(args.output, owner, end):
            started = time.monotonic()
            observe(args.output, cycle)
            for name, model, root in ARMS:
                if name not in args.arms or should_stop(args.output, owner, end):
                    continue
                path = checkpoint_for(name, root)
                if not path.exists():
                    print('WAIT_CHECKPOINT', name, flush=True)
                    continue
                sig = (str(path), path.stat().st_mtime_ns)
                if seen.get(name) == sig:
                    continue
                dest = args.output/f'cycle_{cycle:03d}_{name}'
                # A single extra GPU0 process, batch8, no concurrent watcher workers.
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                           CUBLAS_WORKSPACE_CONFIG=':4096:8')
                with dest.with_suffix('.log').open('x') as log:
                    child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).with_name('interim_evaluation.py')),
                        '--checkpoint', str(path), '--model', model, '--cache', CACHE,
                        '--output', str(dest)], stdout=log, stderr=subprocess.STDOUT,
                        env=env, start_new_session=True)
                    deadline = time.monotonic()+600
                    while child.poll() is None:
                        if should_stop(args.output, owner, end) or time.monotonic() >= deadline:
                            stop_owned(child)
                            break
                        time.sleep(1)
                    rc = child.wait()
                child = None
                if rc == 0:
                    result = json.loads((dest/'result.json').read_text())
                    if result['status'] != 'passed':
                        raise ValueError('Invalid result')
                    print('INTERIM_RESULT '+json.dumps({'arm':name,'cycle':cycle,**result}), flush=True)
                    seen[name] = sig
                else:
                    print('EVALUATION_FAILED', name, rc, dest, flush=True)
                    # Preserve failure; never restart/change the training to fix inference.
                    with dest.with_suffix('.log').open() as log:
                        print(log.read()[-5000:], flush=True)
                    status = 'partial_failure'
            observe(args.output, cycle)
            if args.once:
                break
            cycle += 1
            while time.monotonic()-started < args.interval and not should_stop(args.output, owner, end):
                time.sleep(1)
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if child is not None:
            stop_owned(child)
        # No GPU is owned by this CPU process; child wait guarantees context exit.
        save(args.output/'released.json', {'status':status,'own_evaluator_exited':True,
             'utc':datetime.now(timezone.utc).isoformat()})
    return 0 if status == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
