"""User-authorized pause at the next complete Depth checkpoint.

The immutable training source has no pause callback. Copy its next complete
native state, then ask its verified native supervisor to cancel its own queue
and perform the normal cleanup/burn handoff. Never signal a worker or PID group.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import time
from datetime import datetime, timezone

from scripts.gpu_process import inspect, process, pidfd_open, pidfd_send_signal

WHY_SHA = '87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea'


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    value = dict(value, spec_sha256=WHY_SHA)
    temp = Path(str(path) + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)
    print('WHY_PAUSE', json.dumps(value), flush=True)


def verify_owner(expected):
    live = inspect()
    if live['host'] != expected['host'] or socket.gethostname() != expected['host']:
        raise RuntimeError('Wrong physical node; no signal')
    if [[g['index'], g['uuid'], g['pids']] for g in live['gpus']] != expected['gpu_mapping']:
        raise RuntimeError('Physical GPU ownership changed; no signal')
    for pid, identity in expected['processes'].items():
        if process(int(pid)) != identity:
            raise RuntimeError('Process start/command identity changed; no signal')
    if read('/mnt/local/_gpu_guard/DISABLED')['owner_pid'] != expected['supervisor_pid']:
        raise RuntimeError('Guard owner changed; no signal')
    return live


def copy_state(source, destination, marker):
    """Copy all native files, with byte identity and no shared mutable inodes."""
    if not marker['complete'] or not marker['state_finite'] or len(marker['files']) != 20:
        raise ValueError('Checkpoint is not a complete finite eight-rank state')
    if any((source / name).stat().st_size != size for name, size in marker['files'].items()):
        raise ValueError('Checkpoint member size mismatch')
    required_bytes = sum(p.stat().st_size for p in source.iterdir() if p.is_file())
    if shutil.disk_usage(destination.parent).free < required_bytes + 2 * 1024**3:
        raise RuntimeError('Insufficient free space for independent saved state; no signal')
    shutil.copytree(source, destination)
    hashes = {}
    for p in sorted(source.iterdir()):
        if not p.is_file() or p.is_symlink():
            raise ValueError('Unexpected checkpoint entry')
        other = destination / p.name
        hashes[p.name] = digest(p)
        if digest(other) != hashes[p.name] or p.stat().st_ino == other.stat().st_ino:
            raise ValueError('Pinned checkpoint not an independent exact copy')
    if read(source / 'depth-checkpoint.json') != marker:
        raise ValueError('Checkpoint marker changed while copying')
    return hashes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--request', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    request = read(a.request)
    root = Path(request['depth_root'])
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    if request['spec_sha256'] != WHY_SHA:
        raise ValueError('Why spec mismatch')
    verify_owner(request)
    # Use the immutable experiment's own selector and original identity.
    import sys
    sys.path.insert(0, str(root / 'source'))
    from efficient_llm.a2_depth.resume import select_checkpoint
    from efficient_llm.a2_depth.heal_long import run_identity
    cfg = read(root / 'results/a2_depth_20261031/config/plan.json')
    identity = run_identity(cfg, 'P300')
    trainer = Path(cfg['results_root']) / 'heals/ST1-dclm-P300-S1-12.8B-lr1e-3/trainer'
    initial, rejected = select_checkpoint(trainer, identity, 8)
    if initial is None:
        raise RuntimeError('No initial complete state; leave training untouched')
    initial_step = int(initial.name[11:])
    write(output / 'request.json', dict(request=request, initial_checkpoint=str(initial),
          initial_step=initial_step, rejected=rejected, state='waiting_next_saved_state'))
    while True:
        verify_owner(request)
        candidate, rejected = select_checkpoint(trainer, identity, 8)
        if candidate and int(candidate.name[11:]) > initial_step:
            break
        time.sleep(10)
    marker = read(candidate / 'depth-checkpoint.json')
    pinned = output / candidate.name
    hashes = copy_state(candidate, pinned, marker)
    # Prove native selection can load the pinned copy's recipe/audit continuity.
    selected, rejects = select_checkpoint(output, identity, 8)
    if selected != pinned:
        raise RuntimeError('Pinned state failed native validation; no signal')
    live = verify_owner(request)
    write(output / 'saved_state.json', dict(state='verified_before_cancel',
          checkpoint=str(pinned), original=str(candidate), global_step=marker['global_step'],
          identity=identity, hashes=hashes, inspection=live, rejected=rejects,
          note='Any updates after this save are intentionally discarded on resume.'))
    pid = request['supervisor_pid']
    fd = pidfd_open(pid)
    try:
        verify_owner(request)
        if process(pid) != request['processes'][str(pid)]:
            raise RuntimeError('Supervisor identity changed')
        write(output / 'signal.json', dict(target=pid, signal='SIGTERM',
              at=datetime.now(timezone.utc).isoformat(), authority=request['authority']))
        pidfd_send_signal(fd, signal.SIGTERM)
    finally:
        os.close(fd)
    # Native supervisor alone owns cleanup, GPU-free checks and burn admission.
    terminal = root / 'supervised/supervisor.json'
    while not read(terminal).get('finished_at'):
        time.sleep(5)
    final = read(terminal)
    write(output / 'terminal.json', dict(state='paused', supervisor=final,
          checkpoint=str(pinned), global_step=marker['global_step'],
          handoff_verified=bool(final.get('burn', {}).get('collective_progress_verified'))))
    print('WHY_DEPTH_PAUSE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
