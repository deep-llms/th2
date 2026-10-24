"""Evaluate a matched D control/exit pair, with one independent GPU per model."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_stop_adaptation import read, write
from scripts.stop_eval_suite import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoints', 'parent-recipe', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--fit-start', type=int, required=True)
    args = parser.parse_args()
    checkpoints = read(args.checkpoints)
    if set(checkpoints) != {'D-continue', 'D-DualCE6'}:
        raise ValueError('Expected exactly the authorized dense pair')
    args.output.mkdir(parents=True, exist_ok=False)
    processes, logs = [], []
    try:
        for gpu, (name, checkpoint) in enumerate(checkpoints.items()):
            log = (args.output / (name + '.log')).open('x')
            logs.append(log)
            command = [sys.executable, str(Path(__file__).resolve().parents[1] / 'eval/stop_v2_study.py'),
                       '--checkpoint', checkpoint, '--parent-recipe', str(args.parent_recipe),
                       '--reference-checkpoint', checkpoints['D-continue'],
                       '--checkpoint-step', str(args.step), '--fit-start', str(args.fit_start),
                       '--arm', 'D', '--output', str(args.output / name)]
            processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                             env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '1'}))
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError('Paired STOP evaluation failed; inspect preserved logs')
            time.sleep(1)
        if any(p.returncode for p in processes):
            raise RuntimeError('Paired STOP evaluation failed')
        results = {name: read(args.output / name / 'result.json') for name in checkpoints}
        if any(r['status'] != 'ok' or r['checkpoint_step'] != args.step for r in results.values()):
            raise ValueError('Incomplete or wrong-step evaluation')
        if any(not check['passed'] for r in results.values() for check in r['cached_checks'].values()):
            raise ValueError('Cached decoding numerical check failed')
        write(args.output / 'summary.json', dict(**summarize(results), results=results,
              adaptation_step=args.step, total_step=10000 + args.step,
              head_training_rows=[args.fit_start, args.fit_start + 256],
              calibration_rows=[args.fit_start + 256, args.fit_start + 512]))
    finally:
        # Only our own evaluation children; never GPU-derived PIDs or groups.
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
