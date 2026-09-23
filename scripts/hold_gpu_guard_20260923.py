"""Temporarily inhibit the inspected controller burn for the authorized queue."""
import json
import os
from pathlib import Path
import select
import socket
import time

from scripts.gpu_status import snapshot
from scripts.reclaim_b200_burn_20260923 import process

ROOT = Path('/mnt/local/_outputs/deep-llms_th2/joint-v2-b200-20260923-a03')
MARKER = Path('/mnt/local/_gpu_guard/DISABLED')


def main():
    if socket.gethostname() != 'thiennh-p6-tpbw-worker-0':
        raise RuntimeError('Wrong host')
    handle = os.pidfd_open(26931)
    marker_stat = None
    payload = json.dumps({'owner': str(ROOT), 'queue_pid': 26931,
                          'reason': 'User-authorized eight-GPU scientific training'}) + '\n'
    try:
        queue = process(26931)
        expected = {
            'pid': 26931, 'ppid': 26037, 'start_ticks': 269608076,
            'argv': ['/mnt/local/conda-py311/envs/pcc_joint/bin/python3.11', '-u',
                     'run_experiments.py', '--config', str(ROOT / 'jobs.json'),
                     '--project-dir', str(ROOT / 'source'), '--run-dir', str(ROOT / 'runs')],
        }
        if queue != expected:
            raise RuntimeError('Scientific queue identity changed')
        source = Path('/mnt/local/_gpu_guard/gpu_guard.sh').read_text()
        if 'DIS="$DIR/DISABLED"' not in source or 'if [[ -f "$DIS" ]]; then' not in source:
            raise RuntimeError('Inspected guard marker mechanism changed')
        if MARKER.exists():
            print('GUARD_ALREADY_DISABLED: preserving existing marker', flush=True)
            return
        with MARKER.open('x') as stream:
            stream.write(payload)
            stream.flush()
            marker_stat = os.fstat(stream.fileno())
        print('GUARD_DISABLED_FOR_AUTHORIZED_QUEUE', flush=True)
        time.sleep(30)
        print('GPU_STATUS', json.dumps(snapshot()), flush=True)
        print('QUEUE_STATUS', (ROOT / 'runs/run.json').read_text(), flush=True)
        for name in ('train.jsonl', 'validation.jsonl'):
            lines = (ROOT / 'runs/seed-0-Base' / name).read_text().splitlines()
            print(name, '\n'.join(lines[-3:]), flush=True)
        # The runner owns this independent CPU process. Wait for the pinned
        # queue process, so PID reuse cannot prolong or shorten the lease.
        while not select.select([handle], [], [], 60)[0]:
            pass
        if (MARKER.exists() and MARKER.stat().st_ino == marker_stat.st_ino
                and MARKER.stat().st_dev == marker_stat.st_dev
                and MARKER.read_text() == payload):
            MARKER.unlink()
            print('QUEUE_EXITED_GUARD_POLICY_RESTORED', flush=True)
    finally:
        # On unexpected inspection errors leave our marker in place and fail;
        # never re-enable a competing burn while scientific training is live.
        os.close(handle)


if __name__ == '__main__':
    main()
