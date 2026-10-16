"""Attach four GPU4–7 fits without stopping or changing existing training.

Only the not-yet-executed evaluation entrypoint receives an audited CPU wait
wrapper. Original evaluator bytes, trainers, models, data and schedules stay
unchanged. Existing supervisor remains the single burn owner. On its failure,
the companion promptly terminates only its own child session, then frees4–7.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

PROJECT = Path('/mnt/local/deep-llms_th2')
VISION = '/mnt/local/conda-py311/envs/ce7454_env/bin/python'
CACHE = '/mnt/local/_data/deep-llms_th2/ce7454/cache'
GPUS = [4, 5, 6, 7]


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(pid):
    p = Path('/proc')/str(pid)
    stat = (p/'stat').read_text().rsplit(')', 1)[1].split()
    if stat[0] in ('Z', 'X'):
        raise ProcessLookupError(pid)
    return {'pid': pid, 'start_ticks': int(stat[19]),
            'cmd_sha256': hashlib.sha256((p/'cmdline').read_bytes()).hexdigest()}


def original_running(original, owner):
    try:
        if identity(owner['pid']) != owner:
            return False
        state = json.loads((original/'supervised/run/run.json').read_text())
        return state['status'] == 'running'
    except (OSError, ValueError, KeyError):
        return False


def stop_owned(proc):
    # Only the session created by this controller, never a GPU-derived PGID.
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)


def supervise(proc, original, owner, deadline):
    while proc.poll() is None:
        if not original_running(original, owner):
            raise RuntimeError('Original queue stopped; reclaiming only companion children')
        if time.time() >= deadline:
            raise TimeoutError('Companion deadline')
        time.sleep(1)
    if proc.returncode:
        raise RuntimeError(f'Companion study exited{proc.returncode}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if Path.cwd() != PROJECT or socket.gethostname() != 'thiennh-p6-q359-worker-0':
        raise RuntimeError('Wrong project/host')
    sys.path.insert(0, str(PROJECT))
    from scripts.gpu_status import require_free
    from scripts.train_then_burn import enable_subreaper, clean_owned_children
    enable_subreaper()
    guard = Path('/mnt/local/_gpu_guard/DISABLED')
    marker = guard.read_text()
    lease = json.loads(marker)
    if lease['output'] != str(args.original/'supervised'):
        raise RuntimeError('Guard belongs to another workflow')
    owner = identity(lease['owner_pid'])
    command = (Path('/proc')/str(owner['pid'])/'cmdline').read_bytes()
    if b'scripts.train_then_burn' not in command or str(args.original).encode() not in command:
        raise RuntimeError('Unrecognized supervisor')
    if not original_running(args.original, owner):
        raise RuntimeError('Original queue not running')
    study = args.original/'supervised/run/study'
    if (study/'complete.json').exists() or any(study.glob('*_full512')):
        raise RuntimeError('Attach only while original base fits are active')
    free = require_free(GPUS)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.original/'extension-lease.json').open('x') as f:
        json.dump({'output': str(args.output), 'owner': owner}, f)
    source = Path(__file__).resolve().parent
    code = args.output/'code'
    code.mkdir()
    for src in source.glob('*.py'):
        shutil.copy2(src, code/src.name)
    save(args.output/'source_manifest.json', {p.name: digest(p) for p in code.glob('*.py')})
    save(args.output/'preflight.json', {'gpus_free': free, 'original_owner': owner,
         'guard_sha256': hashlib.sha256(marker.encode()).hexdigest(),
         'started_at': datetime.now(timezone.utc).isoformat()})
    proc = None
    gate_installed = False
    receipt = {'status': 'running'}
    save(args.output/'controller.json', receipt)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Companion signal{signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        # Validate original immutable bytes before the narrow orchestration patch.
        old_code = args.original/'code'
        before = json.loads((args.original/'source_manifest.json').read_text())
        for name, expected in before.items():
            if digest(old_code/name) != expected:
                raise RuntimeError(f'Original frozen source changed:{name}')
        original_eval = old_code/'evaluate_deadline_original.py'
        shutil.copy2(old_code/'evaluate_deadline.py', original_eval)
        shutil.copy2(code/'extension_gate.py', old_code/'extension_gate.py')
        cfg = old_code/'extension_gate.json'
        save(cfg, {'original_evaluator': str(original_eval),
                   'release_receipt': str(args.output/'released.json')})
        wrapper = 'from extension_gate import main\nraise SystemExit(main('+repr(str(cfg))+'))\n'
        tmp = old_code/'evaluate_deadline.attach.tmp'
        tmp.write_text(wrapper)
        if not original_running(args.original, owner) or guard.read_text() != marker:
            raise RuntimeError('Original ownership changed during attachment')
        require_free(GPUS)
        tmp.replace(old_code/'evaluate_deadline.py')
        gate_installed = True
        save(args.output/'gate_patch.json', {'original_evaluator_sha256': before['evaluate_deadline.py'],
             'wrapper_sha256': digest(old_code/'evaluate_deadline.py'),
             'unchanged_training_files': {k: v for k,v in before.items() if k != 'evaluate_deadline.py'},
             'note': 'CPU wait after exact original evaluation. No training/source-model mutation.'})
        with (args.output/'study.log').open('x') as log:
            proc = subprocess.Popen([VISION, '-u', str(code/'band_study.py'),
                '--cache', CACHE, '--output', str(args.output/'study')],
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='4,5,6,7', OMP_NUM_THREADS='1',
                         MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'),
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            receipt['child'] = identity(proc.pid)
            save(args.output/'controller.json', receipt)
            supervise(proc, args.original, owner,
                      datetime.fromisoformat('2026-10-17T02:05:00+00:00').timestamp())
        complete = json.loads((args.output/'study/complete.json').read_text())
        if complete['status'] != 'passed':
            raise RuntimeError('Missing validated completion')
        receipt['status'] = 'passed'
    except BaseException as error:
        receipt.update(status='failed', error=repr(error))
        import traceback
        traceback.print_exc()
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            if proc is not None:
                stop_owned(proc)
            clean_owned_children()
            # CUDA context teardown may briefly outlive process reaping.
            for attempt in range(10):
                try:
                    free = require_free(GPUS)
                    break
                except RuntimeError:
                    if attempt == 9:
                        raise
                    time.sleep(1)
            # Gate can release only AFTER all own CUDA workers are gone.
            save(args.output/'released.json', {'gpus_free': True, 'gpus': GPUS,
                 'snapshot': free, 'study_status': receipt['status']})
        except BaseException as error:
            receipt['release_error'] = repr(error)
        receipt.update(finished_at=datetime.now(timezone.utc).isoformat(), gate_installed=gate_installed)
        save(args.output/'controller.json', receipt)
    return 0 if receipt['status'] == 'passed' and 'release_error' not in receipt else 1


if __name__ == '__main__':
    sys.exit(main())
