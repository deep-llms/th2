"""Original runner-burn ownership checks; only verified worker pidfds are signaled."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import time
from gpu_status import snapshot, require_free

NODE = 'thiennh-p6-8mgy-worker-0'
BURN = '/tmp/llm_pretrain_burn.py'
BURN_SHA = '3cdcc857bd01b096e20a02640fa85f0b8be7607e3c2b22a89a704bbac3650857'


def check(ok, message):
    if not ok:
        raise RuntimeError(message)


def identity(pid):
    check(pid > 1, 'PID 1 is never an eligible process')
    p = Path('/proc')/str(pid)
    fields = (p/'stat').read_text().rsplit(')', 1)[1].split()
    return (int(fields[1]), fields[19], (p/'cmdline').read_bytes().split(b'\0'))


def preflight():
    check(socket.gethostname() == NODE, 'Wrong node')
    check(hashlib.sha256(Path(BURN).read_bytes()).hexdigest() == BURN_SHA, 'Original burn changed')
    check(Path('/mnt/local/_gpu_guard/DISABLED').is_file(), 'GPU guard changed; operator review required')


def inspect(indices, active=False):
    preflight()
    all_gpus = snapshot(list(range(8)))
    selected = [g for g in all_gpus if g['index'] in indices]
    check(len(selected) == len(indices) and len(indices) == len(set(indices)), 'Wrong GPU subset')
    check(all(len(g['pids']) == 1 for g in selected), 'Expected one burn worker per selected GPU')
    pids = [g['pids'][0] for g in selected]
    check(len(set(pids)) == len(pids), 'One process owns multiple selected GPUs')
    check(all(not set(g['pids']) & set(pids) for g in all_gpus if g['index'] not in indices),
          'Selected process also owns an unselected GPU')
    workers = {pid: identity(pid) for pid in pids}
    parents = {x[0] for x in workers.values()}
    check(len(parents) == 1, 'Workers have different launchers')
    parent = parents.pop()
    parent_identity = identity(parent)
    check(BURN.encode() in parent_identity[2], 'Not the original burn launcher')
    check(all(b'--multiprocessing-fork' in x[2] for x in workers.values()), 'Not original spawned workers')
    # mp.spawn reaps sibling workers when one exits. A selected burn group
    # therefore must not have peer workers on an unselected device.
    for g in all_gpus:
        if g['index'] not in indices:
            check(all(identity(pid)[0] != parent for pid in g['pids']),
                  'Burn launcher also owns an unselected GPU; refuse partial group stop')
    if active:
        check(all(g['utilization_percent'] >= 50 for g in selected), 'Burn utilization not established')
    return selected, workers, parent, parent_identity


def stop(indices):
    selected, workers, parent, parent_id = inspect(indices)
    fds = []
    try:
        for pid, expected in workers.items():
            fd = os.pidfd_open(pid)
            fds.append((pid, fd))
            check(identity(pid) == expected, 'Worker identity changed before signaling')
        again, current, parent2, parent_id2 = inspect(indices)
        check(current == workers and parent2 == parent and parent_id2 == parent_id,
              'Ownership changed before signaling')
        for pid, fd in fds:
            try:
                signal.pidfd_send_signal(fd, signal.SIGKILL)
            except ProcessLookupError:
                pass  # Known parent can reap a peer after the first worker exits.
            print('STOPPED_VERIFIED_BURN_WORKER', pid, flush=True)
    finally:
        for _, fd in fds:
            os.close(fd)


def start(indices, session, log, port):
    preflight()
    require_free(indices)
    check(not Path(log).exists(), 'Burn log already exists')
    check(subprocess.run(['tmux', 'has-session', '-t', session], capture_output=True).returncode != 0,
          'Burn session already exists')
    with socket.socket() as s:
        s.bind(('127.0.0.1', port))
    argv = ['env', 'CUDA_VISIBLE_DEVICES='+','.join(map(str, indices)), 'CUDA_DEVICE_ORDER=PCI_BUS_ID', 'MASTER_ADDR=127.0.0.1',
            f'MASTER_PORT={port}', 'NCCL_DEBUG=INFO', '/usr/bin/python3', '-u', BURN]
    command = 'exec '+shlex.join(argv)+' >'+shlex.quote(str(log))+' 2>&1'
    subprocess.run(['tmux', 'new-session', '-d', '-s', session, command], check=True)
    subprocess.run(['tmux', 'set-option', '-w', '-t', session, 'remain-on-exit', 'on'], check=True)
    # Keep stdin attached to tmux. Do not replace this with a detached shell '&'.
    time.sleep(30)
    for attempt in range(10):
        try:
            result = inspect(indices, active=True)
            break
        except RuntimeError:
            if attempt == 9:
                raise
            time.sleep(10)
    time.sleep(10)
    result = inspect(indices, active=True)
    pane = subprocess.check_output(['tmux', 'display-message', '-p', '-t', session, '#{pane_dead}'], text=True)
    check(pane.strip() == '0', 'Burn tmux pane exited')
    print(json.dumps(dict(success=True, session=session, launcher=result[2], gpus=result[0])), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['inspect', 'stop', 'start', 'free'])
    p.add_argument('--gpus', nargs='+', type=int, required=True)
    p.add_argument('--session')
    p.add_argument('--log')
    p.add_argument('--port', type=int)
    a = p.parse_args()
    check(a.gpus and len(set(a.gpus)) == len(a.gpus) and all(0 <= i < 8 for i in a.gpus), 'Invalid GPU indices')
    preflight()
    if a.action == 'stop':
        stop(a.gpus)
    elif a.action == 'start':
        check(a.session and a.log and a.port, 'Start needs session, log and port')
        start(a.gpus, a.session, a.log, a.port)
    elif a.action == 'free':
        print(json.dumps(require_free(a.gpus)), flush=True)
    else:
        result = inspect(a.gpus, active=True)
        print(json.dumps(dict(success=True, launcher=result[2], gpus=result[0])), flush=True)


if __name__ == '__main__':
    main()
