"""Reclaim only freshly verified GPU burn workers, never a launcher or group.

Run only after the user authorizes stopping the named burn. Without --stop,
this performs only ownership verification and prints the exact identities.
"""
import argparse
import json
import os
from pathlib import Path
import signal

from scripts.gpu_status import snapshot


def identity(pid):
    if pid <= 1:
        raise ValueError('PID 1 and invalid PIDs are never eligible')
    root = Path(f'/proc/{pid}')
    stat = (root/'stat').read_text().rsplit(')', 1)[1].split()
    if stat[0] == 'Z':
        raise ProcessLookupError(f'Process {pid} has already exited')
    return dict(pid=pid, parent=int(stat[1]), start=stat[19],
                argv=[v.decode() for v in (root/'cmdline').read_bytes().split(b'\0') if v])


def verified_workers(burn_path, gpus):
    status = snapshot(gpus)
    if not any(gpu['pids'] for gpu in status):
        return [], None
    if any(len(gpu['pids']) != 1 for gpu in status):
        raise RuntimeError('Expected one burn worker per selected GPU')
    workers = [identity(gpu['pids'][0]) for gpu in status]
    if len({p['pid'] for p in workers}) != len(gpus):
        raise RuntimeError('Worker/GPU mapping is not one to one')
    launchers = []
    for worker in workers:
        cursor, seen, candidates = worker, set(), set()
        while cursor['pid'] > 1 and cursor['pid'] not in seen:
            seen.add(cursor['pid'])
            # Exact argv token; never search an embedded shell command string.
            if (burn_path in cursor['argv'] and
                    Path(cursor['argv'][0]).name.startswith('python')):
                candidates.add(cursor['pid'])
            if cursor['parent'] <= 1:
                break
            cursor = identity(cursor['parent'])
        launchers.append(candidates)
    common = set.intersection(*launchers)
    if len(common) != 1:
        raise RuntimeError('GPU workers do not share one verified burn launcher')
    launcher = identity(common.pop())
    if launcher['pid'] in {worker['pid'] for worker in workers}:
        raise RuntimeError('Refusing to signal a burn launcher')
    # No selected worker may own an unselected GPU.
    selected_pids = {p['pid'] for p in workers}
    for gpu in snapshot():
        if gpu['index'] not in gpus and selected_pids.intersection(gpu['pids']):
            raise RuntimeError('Selected workers also own an unselected GPU')
    return workers, launcher


def reclaim(burn_path, gpus, *, stop=False):
    workers, launcher = verified_workers(burn_path, gpus)
    print(json.dumps(dict(workers=workers, launcher=launcher)), flush=True)
    if not stop or not workers:
        return
    fresh, fresh_launcher = verified_workers(burn_path, gpus)
    if workers != fresh or launcher != fresh_launcher:
        raise RuntimeError('Burn identities changed; no signals sent')
    for worker in workers:
        try:
            if identity(worker['pid']) != worker:
                raise RuntimeError('Worker identity changed before signal')
            os.kill(worker['pid'], signal.SIGKILL)
        except (FileNotFoundError, ProcessLookupError):
            # mp.spawn may reap siblings as soon as the first worker exits.
            print('VERIFIED_WORKER_ALREADY_EXITED', worker['pid'], flush=True)
            continue
        print('STOPPED_VERIFIED_BURN_WORKER', worker['pid'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--burn-path', required=True)
    parser.add_argument('--gpus', type=int, nargs='+', required=True)
    parser.add_argument('--stop', action='store_true')
    args = parser.parse_args()
    reclaim(args.burn_path, args.gpus, stop=args.stop)


if __name__ == '__main__':
    main()
