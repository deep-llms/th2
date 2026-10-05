"""One-off, explicitly authorized cleanup of the 2026-10-05 FA4 screen.

Run only as the first CPU job inside the replacement train_then_burn queue.
Source dataset shards and the completed A reference must remain untouched.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket

from scripts.gpu_status import require_free
from scripts.verified_gpu_reclaim import process

OUTPUTS = Path('/mnt/local/_outputs/deep-llms_th2')
OLD = OUTPUTS / 'proxy-fa4-screen-seed42-2500-20261005-a01'
NEW = OUTPUTS / 'proxy-fa4-screen-seed42-2500-20261005-a02'
CONTROL = OUTPUTS / 'proxy-fa4-restart-control-20261005-a01'
DATA = Path('/mnt/local/_data/deep-llms_th2/cx_sampled_old/subsets/qwen3_0.6b_base_en_30B')
ROOTS = [DATA / split for split in ('train', 'validation')]


def inventory(roots):
    caches, sources = [], []
    for root in roots:
        if not root.is_dir() or root.resolve() != root:
            raise RuntimeError(f'Invalid dataset root: {root}')
        for path in sorted(root.rglob('*')):
            if path.is_symlink():
                raise RuntimeError(f'Refusing dataset symlink: {path}')
            if path.is_file():
                stat = path.stat()
                item = dict(path=str(path), bytes=stat.st_size, inode=stat.st_ino, mtime_ns=stat.st_mtime_ns)
                (caches if path.name.startswith(('cache-', 'tmp-')) else sources).append(item)
    return caches, sources


def require_unused(roots, process_dirs=None):
    prefixes = tuple(str(path) + '/' for path in roots)
    for proc in (Path('/proc').glob('[0-9]*') if process_dirs is None else process_dirs):
        try:
            for fd in (proc / 'fd').iterdir():
                try:
                    target = os.readlink(fd)
                except FileNotFoundError:
                    continue
                if target.startswith(prefixes):
                    raise RuntimeError(f'Open file under cleanup roots: {fd}: {target}')
            if any(any(prefix in line for prefix in prefixes) for line in (proc / 'maps').read_text().splitlines()):
                raise RuntimeError(f'Mapped file under cleanup roots: {proc}')
        except (FileNotFoundError, ProcessLookupError):
            continue  # Process exited during the read-only check.
        # Permission failures deliberately fail closed.


def delete_caches(roots, expected_sources):
    caches, sources = inventory(roots)
    if sources != expected_sources:
        raise RuntimeError('Source dataset inventory changed; refusing cleanup')
    require_unused(roots)
    for item in caches:
        path = Path(item['path'])
        stat = path.lstat()
        if path.is_symlink() or (stat.st_ino, stat.st_size, stat.st_mtime_ns) != (item['inode'], item['bytes'], item['mtime_ns']):
            raise RuntimeError(f'Cache identity changed: {path}')
        path.unlink()
    remaining, sources_after = inventory(roots)
    if remaining or sources_after != sources:
        raise RuntimeError('Post-cleanup dataset verification failed')
    return caches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname() != 'thiennh-p6-tjx3-worker-0':
        raise RuntimeError('Wrong machine')
    if args.output != NEW / 'supervised/run/cleanup.json' or args.output.exists():
        raise RuntimeError('Cleanup receipt must be fresh and inside the replacement queue')
    inspection = json.loads((CONTROL / 'inspection.json').read_text())
    stopped = json.loads((CONTROL / 'stop-request.json').read_text())
    if stopped['supervisor'] != inspection['supervisor']:
        raise RuntimeError('Stop receipt identity mismatch')
    try:
        if process(inspection['supervisor']['pid'])['start_ticks'] == inspection['supervisor']['start_ticks']:
            raise RuntimeError('Old supervisor still exists')
    except FileNotFoundError:
        pass
    guard = json.loads(Path('/mnt/local/_gpu_guard/DISABLED').read_text())
    if guard['output'] != str(NEW / 'supervised') or guard['owner_pid'] != os.getppid():
        raise RuntimeError('Cleanup must be owned by the replacement supervisor')
    free = require_free(list(range(8)))
    if inspection['old_output'] != str(OLD) or inspection['dataset_roots'] != list(map(str, ROOTS)):
        raise RuntimeError('Inspection scope differs from authorized paths')
    if OLD.resolve() != OLD or not OLD.is_dir():
        raise RuntimeError('Old output missing or redirected')
    terminal = json.loads((OLD / 'supervised/supervisor.json').read_text())
    if not terminal.get('finished_at') or not terminal.get('burn', {}).get('collective_progress_verified'):
        raise RuntimeError('Old supervisor handoff incomplete')
    require_unused([OLD])
    # Preserve the small stop/handoff audit outside the directory being removed.
    with (CONTROL / 'old-supervisor-terminal.json').open('x') as handle:
        json.dump(terminal, handle, indent=2)
    deleted = delete_caches(ROOTS, inspection['source_files'])
    shutil.rmtree(OLD)  # Exactly the inspected screen; no parent/sibling deletion.
    value = dict(status='cleaned', old_output_removed=str(OLD), deleted_cache_count=len(deleted),
                 deleted_cache_bytes=sum(item['bytes'] for item in deleted), deleted_cache_files=deleted,
                 source_files_preserved=len(inspection['source_files']), gpus_free=free)
    with args.output.open('x') as handle:
        json.dump(value, handle, indent=2)
    print(json.dumps({k:v for k,v in value.items() if k not in ('deleted_cache_files','gpus_free')}), flush=True)


if __name__ == '__main__':
    main()
