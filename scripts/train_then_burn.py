"""Authorized eight-GPU queue with verified burn reclamation and final handoff.

Run in an independent tmux session. Training is the existing sequential runner.
Linux subreaper ownership lets cleanup include orphaned children of THIS queue,
without inferring ownership from a GPU PID or signaling unrelated processes.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import traceback

from run_experiments import load_jobs, now, run_jobs, write_json
from scripts.gpu_status import require_free, snapshot
from scripts.verified_gpu_reclaim import process, pidfd_open, pidfd_send_signal, ownership

GPUS = list(range(8))
BURN_HASH = '2b32968798e2200a8148a3395f1d37ae06e92b6340a74a2f192bfe1a48bcf174'
APPROVED_BURNS = {
    '/tmp/llm_pretrain_burn.py': '3cdcc857bd01b096e20a02640fa85f0b8be7607e3c2b22a89a704bbac3650857',
    '/mnt/local/_gpu_guard/polite_burn.py': '089f55c6b83a01cb5234bb2be5eb3613dd1eca4f65b1aef8a03b2ce44881a379',
}
GUARD_HASH = '3657a891b9e75e1bf57e9cf13d6db24f89ca2b625bbbfb80a19a141d64a1e3b5'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def argv(pid):
    return [x.decode() for x in (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0') if x]


def approved_launcher(pid, approved):
    args = argv(pid)
    # Only a Python script invocation, not an arbitrary command containing a path.
    tail = args[1:]
    while tail and tail[0] in ('-u', '-B'):
        tail = tail[1:]
    return (bool(tail) and Path(process(pid)['exe']).name.startswith('python')
            and tail[0] in approved and digest(tail[0]) == approved[tail[0]])


def stop_burns(output, approved):
    before = snapshot(GPUS)
    workers = sorted({pid for gpu in before for pid in gpu['pids']})
    records, handles = {}, {}
    try:
        for pid in workers:
            record = process(pid)
            launcher = pid if approved_launcher(pid, approved) else record['ppid']
            if launcher <= 1 or not approved_launcher(launcher, approved):
                raise RuntimeError(f'GPU worker {pid} is not an approved burn; no signals sent')
            records[pid] = record
            records[launcher] = process(launcher)
        for pid in workers:
            handles[pid] = pidfd_open(pid)
        if ownership(snapshot(GPUS)) != ownership(before):
            raise RuntimeError('GPU ownership changed before reclamation')
        if any(process(pid) != record for pid, record in records.items()):
            raise RuntimeError('Burn identity changed before reclamation')
        write_json(output / 'reclaim.json', {'at': now(), 'before': before, 'identities': records,
                                            'signal_targets': workers})
        print('AUTHORIZED_BURN_WORKER_STOP', workers, flush=True)
        for pid in workers:
            try:
                pidfd_send_signal(handles[pid], signal.SIGKILL)
            except ProcessLookupError:
                pass  # An earlier worker exit can make mp.spawn stop its siblings.
        time.sleep(30)
        write_json(output / 'gpus-free-before-training.json', {'at': now(), 'gpus': require_free(GPUS)})
        print('ALL_EIGHT_GPUS_VERIFIED_FREE', flush=True)
    finally:
        for fd in handles.values():
            os.close(fd)


def enable_subreaper():
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), 'Cannot own orphaned queue descendants')


def descendants():
    records = {}
    for path in Path('/proc').iterdir():
        if path.name.isdigit() and int(path.name) > 1:
            try:
                item = process(int(path.name))
                records[item['pid']] = item
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass
    owned = {os.getpid()}
    while True:
        added = {pid for pid, item in records.items() if item['ppid'] in owned} - owned
        if not added:
            break
        owned.update(added)
    return {pid: records[pid] for pid in owned if pid != os.getpid()}


def clean_owned_children():
    # Only descendants of this supervisor (including subreaper-adopted orphans).
    # Re-scan to catch children reparented or forked during shutdown.
    for attempt in range(20):
        while True:
            try:
                pid, _ = os.waitpid(-1, os.WNOHANG)
                if pid == 0:
                    break
            except ChildProcessError:
                break
        records = descendants()
        if not records:
            return
        for pid, record in records.items():
            fd = None
            try:
                fd = pidfd_open(pid)
                fresh = descendants().get(pid)
                if fresh is None or fresh['start_ticks'] != record['start_ticks'] or fresh['command_sha256'] != record['command_sha256']:
                    continue
                pidfd_send_signal(fd, signal.SIGTERM if attempt < 5 else signal.SIGKILL)
            except (ProcessLookupError, FileNotFoundError):
                pass
            finally:
                if fd is not None:
                    os.close(fd)
        time.sleep(1)
    raise RuntimeError('Owned training descendants did not exit; refusing a competing burn')


def burn_progress(text):
    ready = {int(m.group(1)) for m in re.finditer(r'gpu_burn_ready rank=(\d+) .*?world_size=8 .*?collective_probe_sum=36(?:\.0)?(?:\s|$)', text)}
    progress = re.findall(r'gpu_burn_progress rank=0 completed_cycles=(\d+) .*?completed_collective_payload_gib=([\d.]+)', text)
    return (ready == set(GPUS) and len(progress) >= 2
            and int(progress[-1][0]) > int(progress[-2][0])
            and float(progress[-1][1]) > float(progress[-2][1]))


def start_burn(output, session, project):
    source = project / 'resources/llm_pretrain_burn.py'
    if digest(source) != BURN_HASH:
        raise RuntimeError('Burn source differs from reviewed source')
    require_free(GPUS)
    # Choose an available local rendezvous port; no other queue is launched here.
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    log = output / 'burn.log'
    if log.exists() or subprocess.run(['tmux', 'has-session', '-t', session], capture_output=True).returncode == 0:
        raise RuntimeError('Refusing to overwrite an existing burn/session')
    settings = {'MATRIX_SIZE': '8192', 'MEMORY_FRACTION': '0.85', 'MIN_FREE_GIB': '8',
                'COMM_TOTAL_MIB': '1137', 'COMM_BUCKET_MIB': '25', 'APPROX_STEP_SECONDS': '0.75',
                'CALIBRATION_GEMMS': '64', 'PROGRESS_EVERY': '10', 'MIN_WORLD_SIZE': '2'}
    env = ['CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7', 'CUDA_DEVICE_ORDER=PCI_BUS_ID',
           'NCCL_NVLS_ENABLE=0', 'PYTHONUNBUFFERED=', 'MASTER_ADDR=127.0.0.1', f'MASTER_PORT={port}']
    env += [f'GPU_BURN_{key}={value}' for key, value in settings.items()]
    # print(..., flush=True) must assemble each readiness line in its buffer;
    # -u splits multi-argument print into interleaving writes across eight ranks.
    command = 'exec ' + shlex.join(['env', *env, sys.executable, str(source)])
    command += ' >' + shlex.quote(str(log)) + ' 2>&1'
    subprocess.run(['tmux', 'new-session', '-d', '-s', session, command], check=True)
    subprocess.run(['tmux', 'set-option', '-w', '-t', session, 'remain-on-exit', 'on'], check=True)
    for _ in range(60):
        time.sleep(5)
        live = subprocess.run(['tmux', 'display-message', '-p', '-t', session, '#{pane_dead}'],
                              capture_output=True, text=True, check=True).stdout.strip()
        if live != '0':
            raise RuntimeError('Burn exited; inspect burn.log')
        if log.exists() and burn_progress(log.read_text()):
            status = snapshot(GPUS)
            workers = {p for gpu in status for p in gpu['pids']}
            if len(workers) != 8 or any(len(gpu['pids']) != 1 for gpu in status):
                raise RuntimeError('Burn worker/GPU mapping differs from eight ranks')
            if any(not approved_launcher(process(p)['ppid'], {str(source): BURN_HASH}) for p in workers):
                raise RuntimeError('Unrecognized GPU worker during burn verification')
            receipt = {'at': now(), 'session': session, 'gpus': status, 'collective_progress_verified': True}
            write_json(output / 'burn-verified.json', receipt)
            return receipt
    raise RuntimeError('Burn did not demonstrate all-rank readiness and advancing collectives')


def execute_queue(jobs, project, output, session):
    receipt = {'started_at': now(), 'training_status': 'running'}
    write_json(output / 'supervisor.json', receipt)
    code = 1
    try:
        code = run_jobs(jobs, project, output / 'run')
        receipt.update(training_status='ok', training_returncode=code)
    except BaseException as error:
        receipt.update(training_status='failed', training_error=repr(error))
        traceback.print_exc()
    finally:
        # A second signal must not interrupt the authorized cleanup/handoff.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            clean_owned_children()
            time.sleep(30)
            receipt['free_after_training'] = require_free(GPUS)
            receipt['burn'] = start_burn(output, session, project)
        except BaseException as error:
            receipt['handoff_error'] = repr(error)
            traceback.print_exc()
            code = 1
        receipt['finished_at'] = now()
        write_json(output / 'supervisor.json', receipt)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inspection', type=Path, required=True)
    parser.add_argument('--host', required=True)
    parser.add_argument('--burn-session', required=True)
    args = parser.parse_args()
    project = Path.cwd().resolve()
    expected = json.loads(args.inspection.read_text())
    status = snapshot(GPUS)
    if socket.gethostname() != args.host or expected['host'] != args.host:
        raise RuntimeError('Node identity changed')
    if [(g['index'], g['uuid']) for g in expected['gpus']] != [(g['index'], g['uuid']) for g in status]:
        raise RuntimeError('GPU identities changed')
    if not all('B200' in g['name'] for g in status):
        raise RuntimeError('Expected eight B200 GPUs')
    jobs = load_jobs(args.config)
    args.output.mkdir(parents=True, exist_ok=False)
    enable_subreaper()
    guard = Path('/mnt/local/_gpu_guard/DISABLED')
    if digest(guard.parent / 'gpu_guard.sh') != GUARD_HASH:
        raise RuntimeError('Guard protocol differs from inspected version')
    marker = json.dumps({'owner_pid': os.getpid(), 'output': str(args.output), 'at': now()})
    with guard.open('x') as handle:
        handle.write(marker)
    inode = guard.stat().st_ino
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Supervisor received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        time.sleep(30)  # Allow any guard pass already in flight to finish.
        approved = {**APPROVED_BURNS, str(project / 'resources/llm_pretrain_burn.py'): BURN_HASH}
        stop_burns(args.output, approved)
        return execute_queue(jobs, project, args.output, args.burn_session)
    finally:
        # Leave the guard held if cleanup/handoff failed: it must not start a
        # competing burn over unverified residual training workers.
        receipt_path = args.output / 'supervisor.json'
        handoff_failed = receipt_path.exists() and 'burn' not in json.loads(receipt_path.read_text())
        # Restore only the marker this process created; never remove another hold.
        if not handoff_failed and guard.exists() and guard.stat().st_ino == inode and guard.read_text() == marker:
            guard.unlink()


if __name__ == '__main__':
    sys.exit(main())
