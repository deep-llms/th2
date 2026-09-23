"""One-time authorized removal of inspected guard workers beside live training."""
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import time

from scripts.gpu_status import snapshot
from scripts.reclaim_b200_burn_20260923 import process

WORKERS = list(range(27017, 27025))
TRAINING = list(range(27540, 27548))
SOURCE = Path('/mnt/local/_gpu_guard/polite_burn.py')
SOURCE_HASH = '089f55c6b83a01cb5234bb2be5eb3613dd1eca4f65b1aef8a03b2ce44881a379'


def validate(status, parent, workers, training, host, digest):
    if host != 'thiennh-p6-tpbw-worker-0' or digest != SOURCE_HASH:
        raise ValueError('Host or guard source changed')
    if parent != {'pid': 26949, 'ppid': 1, 'start_ticks': 269608218,
                  'argv': ['/usr/bin/python3', str(SOURCE)]}:
        raise ValueError('Guard launcher identity changed')
    expected = [(i, sorted([w, t])) for i, (w, t) in enumerate(zip(WORKERS, TRAINING))]
    if [(g['index'], sorted(g['pids'])) for g in status] != expected:
        raise ValueError('GPU ownership changed')
    if len(workers) != 8 or len(training) != 8:
        raise ValueError('Incomplete worker inspection')
    for i, record in enumerate(workers):
        expected_record = {
            'pid': WORKERS[i], 'ppid': 26949, 'start_ticks': 269608372,
            'argv': ['/usr/bin/python3', '-B', '-c',
                     'from multiprocessing.spawn import spawn_main; '
                     f'spawn_main(tracker_fd=37, pipe_handle={39 + 2*i})',
                     '--multiprocessing-fork'],
        }
        if record != expected_record:
            raise ValueError('Guard worker identity changed')
    for pid, record in zip(TRAINING, training):
        if (record['pid'] != pid or record['ppid'] != 26941
                or record['start_ticks'] != 269609058
                or record['argv'][:5] != [
                    '/mnt/local/conda-py311/envs/pcc_joint/bin/python3.11',
                    '-u', '-m', 'pcc.joint', 'train']):
            raise ValueError('Scientific worker identity changed')


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorized-stop', action='store_true', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Receipt already exists')
    handles = []
    try:
        for pid in WORKERS:
            handles.append(os.pidfd_open(pid))
        workers = [process(pid) for pid in WORKERS]
        training = [process(pid) for pid in TRAINING]
        validate(snapshot(), process(26949), workers, training,
                 socket.gethostname(), hashlib.sha256(SOURCE.read_bytes()).hexdigest())
        print(json.dumps({'authorized_guard_workers': workers}), flush=True)
        for handle in handles:
            try:
                signal.pidfd_send_signal(handle, signal.SIGKILL)
            except ProcessLookupError:
                pass  # Pinned worker exited after its NCCL peer; no PID reuse.
        time.sleep(30)
        after = snapshot()
        if [(g['index'], g['pids']) for g in after] != list(enumerate([[p] for p in TRAINING])):
            raise RuntimeError('Unexpected remaining GPU ownership; inspect, never escalate')
        if [process(pid) for pid in TRAINING] != training:
            raise RuntimeError('Scientific worker identities changed')
        with args.output.open('x') as handle:
            json.dump({'status': 'ok', 'stopped_workers': workers,
                       'preserved_training': training, 'after': after}, handle, indent=2)
        print('GUARD_STOPPED_TRAINING_PRESERVED', flush=True)
    finally:
        for handle in handles:
            os.close(handle)


if __name__ == '__main__':
    main()
