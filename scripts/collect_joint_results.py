"""Inspect GPU ownership and bundle small joint-training results; never signal GPUs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import tarfile
import time

from scripts.gpu_status import query, snapshot


def inspect():
    result = {'time': datetime.now(timezone.utc).isoformat(), 'gpus': snapshot(),
              'power': query('index,power.draw,power.limit', 'gpu'), 'processes': {}}
    for gpu in result['gpus']:
        for worker in gpu['pids']:
            for pid in [worker]:
                while pid > 1 and str(pid) not in result['processes']:
                    root = Path('/proc') / str(pid)
                    try:
                        fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
                        argv = [s.decode(errors='replace') for s in
                                (root / 'cmdline').read_bytes().split(b'\0') if s]
                        record = {'pid': pid, 'ppid': int(fields[1]),
                                  'start_ticks': int(fields[19]),
                                  'cpu_ticks': int(fields[11]) + int(fields[12]),
                                  'burn_launcher': argv == ['/usr/bin/python3', '/mnt/local/_gpu_guard/polite_burn.py'],
                                  'spawn_worker': bool(argv and argv[0] == '/usr/bin/python3'
                                                       and '--multiprocessing-fork' in argv
                                                       and any('from multiprocessing.spawn import spawn_main;' in s for s in argv))}
                        result['processes'][str(pid)] = record
                        pid = record['ppid']
                    except (FileNotFoundError, ProcessLookupError):
                        result['processes'][str(pid)] = {'pid': pid, 'exited_during_inspection': True}
                        break
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    queue = json.loads((root / 'runs/complete.json').read_text())
    if queue['status'] != 'ok' or len(queue['jobs']) != 7 or any(j['status'] != 'ok' for j in queue['jobs']):
        raise ValueError('Six experiments and report have not all completed')
    first = inspect()
    time.sleep(10)
    second = inspect()
    guard = Path('/mnt/local/_gpu_guard')
    source_hash = hashlib.sha256((guard / 'polite_burn.py').read_bytes()).hexdigest()
    verified = (not (guard / 'DISABLED').exists() and source_hash ==
                '089f55c6b83a01cb5234bb2be5eb3613dd1eca4f65b1aef8a03b2ce44881a379')
    for sample in (first, second):
        verified &= [g['index'] for g in sample['gpus']] == list(range(8))
        for gpu in sample['gpus']:
            if len(gpu['pids']) != 1 or gpu['utilization_percent'] <= 0:
                verified = False
                continue
            worker = sample['processes'].get(str(gpu['pids'][0]), {})
            parent = sample['processes'].get(str(worker.get('ppid')), {})
            verified &= worker.get('spawn_worker', False) and parent.get('burn_launcher', False)
    evidence = {'host': socket.gethostname(), 'burn_verified_on_all_eight_gpus': bool(verified),
                'guard_disabled': (guard / 'DISABLED').exists(), 'source_sha256': source_hash,
                'samples': [first, second]}
    evidence_path = args.output / 'gpu-burn-verification.json'
    evidence_path.write_text(json.dumps(evidence, indent=2) + '\n')
    files = [root / name for name in ('config.json', 'jobs.json', 'capacity_ready.json',
                                      'runs/run.json', 'runs/complete.json')]
    files += [root / f'capacity-{arm}/capacity.json' for arm in ('Base', 'Shallow', 'Deep')]
    for seed in range(2):
        for arm in ('Base', 'Shallow', 'Deep'):
            files += [root / f'runs/seed-{seed}-{arm}' / name for name in
                      ('complete.json', 'identity.json', 'train.jsonl', 'validation.jsonl', 'eval.npz')]
    files += sorted((root / 'runs/report').iterdir())
    entries = [(path, str(path.relative_to(root))) for path in files]
    entries.append((evidence_path, 'gpu-burn-verification.json'))
    manifest = []
    for path, name in entries:
        if (path.is_symlink() or not path.is_file() or path.stat().st_size > 25_000_000
                or path.suffix not in {'.json', '.jsonl', '.md', '.csv', '.png', '.npz'}):
            raise ValueError(f'Unexpected or oversized result: {name}')
        manifest.append({'path': name, 'bytes': path.stat().st_size,
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest_path = args.output / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    archive = args.output / 'results.tar.gz'
    with tarfile.open(archive, 'x:gz') as bundle:
        for path, name in entries:
            bundle.add(path, arcname=name, recursive=False)
        bundle.add(manifest_path, arcname='manifest.json', recursive=False)
    if archive.stat().st_size > 25_000_000:
        raise ValueError('Archive exceeds runner export limit')
    receipt = {'status': 'ok', 'archive': archive.name, 'bytes': archive.stat().st_size,
               'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'files': len(entries),
               'burn_verified_on_all_eight_gpus': bool(verified)}
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2), flush=True)
    print(json.dumps(evidence, indent=2), flush=True)


if __name__ == '__main__':
    main()
