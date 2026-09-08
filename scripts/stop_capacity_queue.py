"""Stop a specifically identified Stagewise queue and its GPU workers; no groups.

Requires explicit queue PID/start-time from a fresh read-only ancestry report.
No deletion, restart, PID1 signaling or name-based process matching.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import time
from scripts.gpu_status import snapshot, require_free
from scripts.reclaim_verified_burn import identity


def unchanged(record):
    """Allow reparenting after the queue exits, never PID reuse or changed argv."""
    current = identity(record['pid'])
    return all(current[key] == record[key] for key in ('pid', 'start', 'argv'))


def identify(queue_pid, queue_start, run_root):
    queue = identity(queue_pid)
    expected = ['bash', 'scripts/train_capacity_b200.sh', run_root, 'B0', 'A128', 'A256', 'A512', 'C', 'D']
    if queue['start'] != queue_start or queue['argv'] != expected:
        raise RuntimeError('Queue identity does not match the authorized run')
    status = snapshot()
    if [gpu['index'] for gpu in status] != list(range(8)) or any(len(gpu['pids']) != 1 for gpu in status):
        raise RuntimeError('Expected eight GPUs with exactly one current training worker each')
    workers = [identity(gpu['pids'][0]) for gpu in status]
    if len({worker['pid'] for worker in workers}) != 8 or len({worker['parent'] for worker in workers}) != 1:
        raise RuntimeError('Training worker mapping changed')
    launcher = identity(workers[0]['parent'])
    if launcher['parent'] != queue_pid or 'launch' not in launcher['argv'] or (
            '/mnt/local/conda-py311/envs/swt/bin/accelerate' not in launcher['argv']):
        raise RuntimeError('GPU launcher does not belong to the identified queue')
    for worker in workers:
        argv = worker['argv']
        if argv[:3] != ['/mnt/local/conda-py311/envs/swt/bin/python3.11', '-u', 'train.py']:
            raise RuntimeError('GPU process is not an expected Stagewise training worker')
        output = argv[argv.index('--output_dir')+1]
        if str(Path(output).parent) != run_root or Path(output).name not in expected[3:]:
            raise RuntimeError('Training output does not belong to the authorized run')
    return queue, launcher, workers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-pid', type=int, required=True)
    parser.add_argument('--queue-start', required=True)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--stop', action='store_true')
    args = parser.parse_args()
    records = identify(args.queue_pid, args.queue_start, args.run_root)
    print('VERIFIED_STOP_TARGETS', json.dumps(records), flush=True)
    if not args.stop:
        return
    if identify(args.queue_pid, args.queue_start, args.run_root) != records:
        raise RuntimeError('Training identities changed; no signals sent')
    queue, launcher, workers = records
    # Disable the dedicated experiment queue first, so it cannot advance to
    # another arm or a burn while workers are being reclaimed. Never touch its
    # parent commands.sh shell, runner tmux server, sleeper, or process group.
    if not unchanged(queue):
        raise RuntimeError('Queue changed before signaling')
    os.kill(queue['pid'], signal.SIGKILL)
    print('STOPPED_EXACT_EXPERIMENT_QUEUE', queue['pid'], flush=True)
    for worker in workers:
        try:
            if not unchanged(worker):
                raise RuntimeError('Worker changed before signaling')
            os.kill(worker['pid'], signal.SIGKILL)
            print('STOPPED_VERIFIED_TRAINING_WORKER', worker['pid'], flush=True)
        except (ProcessLookupError, FileNotFoundError):
            print('VERIFIED_WORKER_ALREADY_EXITED', worker['pid'], flush=True)
    time.sleep(30)
    require_free(list(range(8)))
    for record in (queue, launcher, *workers):
        try:
            if unchanged(record):
                raise RuntimeError(f'Owned process has not exited: {record["pid"]}')
        except (ProcessLookupError, FileNotFoundError):
            pass
    print('STAGEWISE_QUEUE_STOPPED_ALL_GPUS_FREE_OUTPUTS_PRESERVED', flush=True)


if __name__ == '__main__':
    main()
