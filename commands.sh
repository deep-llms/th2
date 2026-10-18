#1 +30+a
#th2-q359-dot-project-cleanup-20261018-a01
set -euo pipefail
python3 - <<'PY'
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

ROOT = Path('/mnt/local/deep-llms_th2')
BURN = ROOT / 'resources/llm_pretrain_burn.py'
BURN_SHA256 = '2b32968798e2200a8148a3395f1d37ae06e92b6340a74a2f192bfe1a48bcf174'
KEEP = {'commands.sh', '_run_log_', '_dl_active.txt', '_install_active.txt',
        '_previous_run_status.log', 'resources'}
REMOVE = {'.gitignore', 'AGENTS.md', 'README.md', 'artifacts',
          'baseline_a_fa4.b200.json', 'ce7454', 'deep_kv', 'deep_kv.b200.json',
          'docs', 'envs', 'eval', 'jobs.example.json', 'prepare_data.py',
          'project.example.json', 'proxy_heads.12layer.b200.json',
          'proxy_heads.b200.json', 'requirements.txt', 'run_experiments.py',
          'scripts', 'template_validation', 'tests', 'train.py'}

def emit(event, **values):
    print(event + ' ' + json.dumps(values, sort_keys=True), flush=True)

def process(pid):
    proc = Path('/proc') / str(pid)
    fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
    args = (proc / 'cmdline').read_bytes().decode().split('\0')[:-1]
    return {'pid': pid, 'ppid': int(fields[1]), 'start_ticks': int(fields[19]),
            'args': args}

def gpu_snapshot():
    query = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
        '--format=csv,noheader,nounits'], text=True)
    result = []
    for row in csv.reader(query.splitlines()):
        if not row:
            continue
        pid, uuid = int(row[0].strip()), row[1].strip()
        worker = process(pid)
        ancestor = worker
        seen = set()
        while str(BURN) not in ancestor['args']:
            if ancestor['pid'] <= 1 or ancestor['pid'] in seen:
                raise RuntimeError('Non-burn GPU workload found; no cleanup authorized for its files')
            seen.add(ancestor['pid'])
            ancestor = process(ancestor['ppid'])
        if (ancestor['pid'] <= 1 or len(ancestor['args']) < 2
                or ancestor['args'][1] != str(BURN)
                or not Path(ancestor['args'][0]).name.startswith('python')):
            raise RuntimeError('Unrecognized burn launcher')
        result.append({'pid': pid, 'gpu_uuid': uuid,
                       'start_ticks': worker['start_ticks'],
                       'launcher_pid': ancestor['pid'],
                       'launcher_start_ticks': ancestor['start_ticks']})
    return sorted(result, key=lambda x: (x['gpu_uuid'], x['pid']))

def guard_tree(path):
    # rmtree does not follow symlinks; refuse nested mounts before any deletion.
    if path.is_symlink():
        return
    if path.is_mount():
        raise RuntimeError('Refusing mounted cleanup path: ' + str(path))
    if path.is_dir():
        def fail(error):
            raise error
        for parent, dirs, _ in os.walk(path, followlinks=False, onerror=fail):
            for name in dirs:
                child = Path(parent) / name
                if not child.is_symlink() and child.is_mount():
                    raise RuntimeError('Refusing nested mount: ' + str(child))

def remove_entry(path):
    if path.is_symlink() or not path.is_dir():
        path.unlink()
    else:
        shutil.rmtree(path)

def main():
    if socket.gethostname() != 'thiennh-p6-q359-worker-0':
        raise RuntimeError('Unexpected machine')
    if ROOT.is_symlink() or ROOT.resolve() != ROOT or Path.cwd().resolve() != ROOT:
        raise RuntimeError('Unexpected project directory')
    if not shutil.rmtree.avoids_symlink_attacks:
        raise RuntimeError('This Python cannot safely remove directory trees')
    if BURN.is_symlink() or hashlib.sha256(BURN.read_bytes()).hexdigest() != BURN_SHA256:
        raise RuntimeError('Unexpected burn source; leave all files unchanged')
    if (ROOT / 'resources').is_symlink():
        raise RuntimeError('Unexpected resources symlink')
    command = ROOT / 'commands.sh'
    if command.is_symlink() or not command.is_file():
        raise RuntimeError('Expected regular commands.sh')
    command_sha = hashlib.sha256(command.read_bytes()).hexdigest()
    names = {p.name for p in ROOT.iterdir()}
    if names - KEEP - REMOVE:
        raise RuntimeError('Unexpected top-level entries: ' + repr(sorted(names - KEEP - REMOVE)))
    targets = [ROOT / name for name in sorted(names & REMOVE)]
    targets += sorted((p for p in (ROOT / 'resources').iterdir() if p != BURN), key=str)
    for path in targets:
        guard_tree(path)

    # Refuse other running Python scripts depending on the code being removed.
    # Never print raw commands/environment or signal any process.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            info = process(int(proc.name))
            cwd = (proc / 'cwd').resolve(strict=True)
            for arg in info['args'][1:]:
                if not arg.endswith('.py') or '\n' in arg:
                    continue
                candidate = Path(arg) if arg.startswith('/') else cwd / arg
                candidate = candidate.resolve()
                if candidate != BURN and candidate.is_relative_to(ROOT):
                    raise RuntimeError('Live project script: PID ' + proc.name + ', ' + str(candidate))
        except (FileNotFoundError, ProcessLookupError):
            continue

    before = gpu_snapshot()
    emit('CLEANUP_PREFLIGHT', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         hostname=socket.gethostname(), project_dir=str(ROOT),
         commands_sha256=command_sha, gpu_processes=before,
         removing=[str(p.relative_to(ROOT)) for p in targets],
         preserving=sorted(KEEP), preserved_resource=BURN.name)
    if gpu_snapshot() != before:
        raise RuntimeError('GPU workload changed during preflight')
    for path in targets:
        remove_entry(path)
        emit('REMOVED', path=str(path.relative_to(ROOT)))

    remaining = sorted(p.name for p in ROOT.iterdir())
    resources = sorted(p.name for p in (ROOT / 'resources').iterdir())
    if set(remaining) != names & KEEP or resources != [BURN.name]:
        raise RuntimeError('Unexpected cleanup result')
    if hashlib.sha256(command.read_bytes()).hexdigest() != command_sha:
        raise RuntimeError('commands.sh changed during cleanup')
    if hashlib.sha256(BURN.read_bytes()).hexdigest() != BURN_SHA256:
        raise RuntimeError('Burn source changed during cleanup')
    time.sleep(10)
    after = gpu_snapshot()
    if after != before:
        raise RuntimeError('GPU ownership changed; inspect before claiming healthy handoff')
    emit('CLEANUP_RESULT', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         remaining=remaining, resources=resources, removed_entries=len(targets),
         gpu_processes_unchanged=True, gpu_processes=after,
         commands_sha256=command_sha, burn_sha256=BURN_SHA256)
    print('DOT_TH2_PROJECT_CLEANUP_OK', flush=True)

if __name__ == '__main__':
    main()
PY
