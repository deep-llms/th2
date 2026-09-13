"""Run-specific GPU safety checks for the authorized 2026-09-13 8mgy smoke.

Not a general kill utility. Stop only the exact original burn workers observed
in the read-only preflight; never a launcher, PID 1, or a process group.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
from gpu_status import snapshot


def check(condition, message='GPU identity/safety check failed'):
    if not condition:
        raise RuntimeError(message)


def identity(pid):
    if pid <= 1:
        raise RuntimeError('PID 1 is never a signal target')
    proc = Path('/proc')/str(pid)
    fields = (proc/'stat').read_text().rsplit(')', 1)[1].split()
    return int(fields[1]), fields[19], (proc/'cmdline').read_bytes().split(b'\0')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['stop-verified-original', 'verify-burn'])
    args = p.parse_args()
    check(socket.gethostname() == 'thiennh-p6-8mgy-worker-0', 'Wrong node')
    burn = Path('/tmp/llm_pretrain_burn.py')
    check(hashlib.sha256(burn.read_bytes()).hexdigest() == '3cdcc857bd01b096e20a02640fa85f0b8be7607e3c2b22a89a704bbac3650857')
    status = snapshot(list(range(8)))
    check(all('B200' in g['name'] and len(g['pids']) == 1 for g in status), 'Unexpected GPU/workload layout')
    pids = [g['pids'][0] for g in status]
    check(len(set(pids)) == 8)
    if args.action == 'verify-burn':
        parents = {identity(pid)[0] for pid in pids}
        check(len(parents) == 1)
        parent = parents.pop()
        argv = identity(parent)[2]
        check(b'/tmp/llm_pretrain_burn.py' in argv)
        check(all(g['utilization_percent'] >= 50 for g in status), 'Burns not consistently active')
        print(json.dumps(dict(success=True, launcher_pid=parent, gpus=status)), flush=True)
        return
    check(Path('/mnt/local/_gpu_guard/DISABLED').is_file(), 'GPU guard state changed; re-inspect')
    check(pids == list(range(496, 504)), 'GPU worker identities changed; re-inspect')
    parent = identity(429)
    check(parent[:2] == (1, '179777866') and b'/tmp/llm_pretrain_burn.py' in parent[2])
    # pidfds bind the signal to the observed process, preventing PID-reuse races.
    fds = []
    try:
        for pid in pids:
            fd = os.pidfd_open(pid)
            fds.append(fd)
            info = identity(pid)
            check(info[:2] == (429, '179777986') and b'--multiprocessing-fork' in info[2])
        check([g['pids'] for g in snapshot(list(range(8)))] == [[pid] for pid in pids])
        for pid, fd in zip(pids, fds):
            try:
                signal.pidfd_send_signal(fd, signal.SIGKILL)
                print('STOPPED_VERIFIED_BURN_WORKER', pid, flush=True)
            except ProcessLookupError:
                # The known burn parent can reap peers after the first exits.
                print('VERIFIED_BURN_WORKER_ALREADY_EXITED', pid, flush=True)
    finally:
        for fd in fds:
            os.close(fd)


if __name__ == '__main__':
    main()
