"""One-time, user-authorized stop of the inspected B200 burn workers only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import time

from scripts.gpu_status import require_free, snapshot

HOST = 'thiennh-p6-tpbw-worker-0'
BURN_HASH = '3cdcc857bd01b096e20a02640fa85f0b8be7607e3c2b22a89a704bbac3650857'
WORKERS = list(range(498, 506))


def process(pid):
    root = Path('/proc') / str(pid)
    fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
    return {'pid': pid, 'ppid': int(fields[1]), 'start_ticks': int(fields[19]),
            'argv': [s.decode() for s in (root / 'cmdline').read_bytes().split(b'\0') if s]}


def validate(status, parent, workers, host, digest):
    if host != HOST or digest != BURN_HASH:
        raise ValueError('Host or burn source differs from the approved inspection')
    if parent != {'pid': 431, 'ppid': 1, 'start_ticks': 258980356,
                  'argv': ['/usr/bin/python3', '/tmp/llm_pretrain_burn.py']}:
        raise ValueError('Burn launcher identity changed')
    if [(g['index'], g['pids']) for g in status] != [(i, [p]) for i, p in enumerate(WORKERS)]:
        raise ValueError('GPU process ownership changed; refusing to stop anything')
    if len(workers) != 8:
        raise ValueError('Incomplete worker inspection')
    for expected_pid, record in zip(WORKERS, workers):
        if (record['pid'] != expected_pid or record['ppid'] != 431 or record['start_ticks'] != 258980483
                or record['argv'][0] != '/usr/bin/python3'
                or '--multiprocessing-fork' not in record['argv']
                or not any('from multiprocessing.spawn import spawn_main;' in arg for arg in record['argv'])):
            raise ValueError('Worker identity changed; refusing to stop anything')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorized-stop', action='store_true', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Stop receipt already exists')
    handles = []
    try:
        # Pin the inspected process identities in the kernel. A PID being
        # reused later cannot redirect pidfd_send_signal to another workload.
        for pid in WORKERS:
            handles.append(os.pidfd_open(pid))
        status = snapshot()
        parent = process(431)
        workers = [process(pid) for pid in WORKERS]
        digest = hashlib.sha256(Path('/tmp/llm_pretrain_burn.py').read_bytes()).hexdigest()
        validate(status, parent, workers, socket.gethostname(), digest)
        # All validation completes before the first signal; never signal the
        # launcher, PID 1, or a process group.
        print(json.dumps({'authorized_workers': workers, 'source_sha256': digest}), flush=True)
        for handle in handles:
            try:
                signal.pidfd_send_signal(handle, signal.SIGKILL)
            except ProcessLookupError:
                # The same verified worker may already have exited when its
                # NCCL peer died. The pinned handle cannot target a reused PID.
                pass
        time.sleep(30)
        after = require_free(list(range(8)))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as handle:
            json.dump({'status': 'ok', 'stopped_workers': workers, 'after': after}, handle, indent=2)
        print('VERIFIED_BURN_STOPPED', flush=True)
    finally:
        for handle in handles:
            os.close(handle)


if __name__ == '__main__':
    main()
