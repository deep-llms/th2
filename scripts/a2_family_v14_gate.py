"""F6 must finish and its report/backup verify before the old Depth bridge proceeds.

Standalone stdlib code: copied outside frozen training source. No process signals,
GPU allocation, networking, or implicit retry. Missing readiness waits; invalid
readiness/completion fails closed and leaves the original Depth checkpoint alone.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SPEC_SHA = '04233f076e8d82b6c76a33178d2944e621a278e01b779b3dce584b2b8ad464aa'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_lock(item):
    path = Path(item['path'])
    if sha(path) != item['sha256']:
        raise ValueError('F6 locked artifact changed: ' + str(path))
    return json.loads(path.read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def verify_ready(ready):
    if ready['spec_sha256'] != SPEC_SHA:
        raise ValueError('Wrong F6 preregistration')
    manifest = read_lock(ready['source_manifest'])
    source = Path(ready['source'])
    for name, digest in manifest['files'].items():
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts or sha(source / rel) != digest:
            raise ValueError('F6 source identity changed: ' + name)
    required = {'scripts/a2_family_f6_run.py', 'docs/A2_family_spec_v1.4.md'}
    if not required.issubset(manifest['files']):
        raise ValueError('Missing F6 coordinator/spec in frozen source')
    if sha(source / 'docs/A2_family_spec_v1.4.md') != SPEC_SHA:
        raise ValueError('F6 spec bytes differ')
    config = read_lock(ready['config'])
    if config['spec_sha256'] != SPEC_SHA:
        raise ValueError('Wrong F6 config')
    return source, config


def verify_completion(receipt):
    value = json.loads(Path(receipt).read_text())
    if value.get('spec_sha256') != SPEC_SHA or not all(value.get(k) is True for k in
            ('execution_complete', 'report_complete', 'backup_verified')):
        raise ValueError('F6/report/backup incomplete; Depth remains deferred')
    required = {'ST1-dclm-S72-S1-0.2B-lr1e-3'} | {
        f'ST1-{ds}-{arm}-S1-0.8B-lr1e-3'
        for ds in ('dclm', 'fwe') for arm in ('S72', 'S188')}
    selected = value['extension_dataset']
    if selected is not None:
        if selected not in ('dclm', 'fwe'):
            raise ValueError('Invalid F6 extension dataset')
        required |= {f'ST1-{selected}-{arm}-S1-3.2B-lr1e-3' for arm in ('S72', 'S188')}
    events = value['events']
    names = [event['name'] for event in events]
    if len(names) != len(set(names)) or set(names) != required:
        raise ValueError('F6 registered coverage differs')
    # Failed scientific/model runs may be reported, never silently omitted.
    if any(not isinstance(e.get('returncode'), int) for e in events):
        raise ValueError('Missing F6 execution outcome')
    if not value.get('backup_files'):
        raise ValueError('No F6 scientific backup')
    for item in [value['report'], value['decision'], *value['backup_files']]:
        if sha(item['path']) != item['sha256']:
            raise ValueError('F6 output hash mismatch')
    return value


def run_gate(root, poll_seconds=30):
    root = Path(root)
    with (root / 'gate.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ready_path = root / 'ready.json'
        while not ready_path.is_file():
            write(root / 'gate-status.json', dict(spec_sha256=SPEC_SHA,
                state='waiting_for_F6_preparation', depth_resume_allowed=False))
            print('FAMILY_V14_WAITING_FOR_F6', flush=True)
            time.sleep(poll_seconds)
        ready = json.loads(ready_path.read_text())
        ready_hash = sha(ready_path)
        source, config = verify_ready(ready)
        receipt = root / 'f6-complete.json'
        attempt = root / 'attempt.json'
        if attempt.exists():
            original = json.loads(attempt.read_text())
            if original['ready_sha256'] != ready_hash or not receipt.is_file():
                raise ValueError('Existing F6 attempt; no blind relaunch')
        else:
            if receipt.exists():
                raise ValueError('Unexpected preexisting F6 receipt')
            write(attempt, dict(spec_sha256=SPEC_SHA, ready_sha256=ready_hash,
                bridge_pid=os.getpid(), config=ready['config']))
            command = [sys.executable, '-u', '-m', 'scripts.a2_family_f6_run',
                '--config', ready['config']['path'], '--receipt', str(receipt)]
            write(root / 'gate-status.json', dict(spec_sha256=SPEC_SHA,
                state='running_F6', depth_resume_allowed=False, command=command))
            result = subprocess.run(command, cwd=source,
                env={**os.environ, 'PYTHONPATH': str(source)})
            if result.returncode:
                raise RuntimeError('F6 coordinator failed; Depth remains deferred')
        verify_completion(receipt)
        write(root / 'gate-status.json', dict(spec_sha256=SPEC_SHA,
            state='F6_report_backup_verified', depth_resume_allowed=True,
            receipt=dict(path=str(receipt), sha256=sha(receipt))))
