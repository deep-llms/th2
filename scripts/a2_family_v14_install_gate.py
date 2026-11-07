"""Install only the not-yet-started Depth bridge hook for authorized Family F6."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket

from scripts.a2_family_v14_gate import SPEC_SHA, sha, read_lock, write
from scripts.gpu_process import inspect, process
from scripts.train_then_burn import argv


def bridge_bytes(original, gate_path, gate_sha, transition):
    text = original.decode()
    anchor = '    record,plan,resume=validate(cfg);require_free(list(range(8)))'
    if text.count(anchor) != 1:
        raise ValueError('Unexpected original bridge entry point')
    hook = ("    # Authorized v1.4 F6 interposition; original training code remains frozen.\n"
        "    import hashlib, runpy\n"
        f"    gate_path = Path({str(gate_path)!r})\n"
        f"    if hashlib.sha256(gate_path.read_bytes()).hexdigest() != {gate_sha!r}:\n"
        "        raise ValueError('Family v1.4 handoff code changed')\n"
        f"    runpy.run_path(str(gate_path))['run_gate']({str(transition)!r})\n")
    return text.replace(anchor, hook + anchor).encode()


def verify_owner(plan):
    live = inspect()
    if live['host'] != plan['host'] or len(live['workers']) != 8:
        raise ValueError('Wrong host or expected live Family workers missing')
    if [g['uuid'] for g in live['gpus']] != plan['gpu_uuids']:
        raise ValueError('GPU identity changed')
    coordinator, supervisor = plan['coordinator'], plan['supervisor']
    if process(coordinator['pid']) != coordinator or process(supervisor['pid']) != supervisor:
        raise ValueError('Family coordinator/supervisor identity changed')
    guard = json.loads(Path('/mnt/local/_gpu_guard/DISABLED').read_text())
    if guard['owner_pid'] != supervisor['pid']:
        raise ValueError('Native guard changed')
    for pid in live['workers']:
        worker = process(pid); launcher = process(worker['ppid'])
        if launcher['ppid'] != coordinator['pid'] or 'efficient_llm.a2_family.heal' not in argv(pid):
            raise ValueError('Worker is not an owned Family heal')
    # A bridge already running would have loaded its old code; refuse such a race.
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit(): continue
        try:
            if 'scripts.a2_family_depth_bridge' in argv(int(entry.name)):
                raise ValueError('Depth bridge already started')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return live


def install(plan, project):
    if socket.gethostname() != plan['host'] or plan['spec_sha256'] != SPEC_SHA:
        raise ValueError('Wrong node/spec')
    root = Path(plan['family_root']); transition = Path(plan['transition_root'])
    if transition.exists(): raise ValueError('Transition already exists; retrieve receipt first')
    launch = read_lock(plan['launch']); cfg = read_lock(launch['config'])
    read_lock(launch['jobs'])
    if cfg['spec_sha256'] != plan['original_spec_sha256']:
        raise ValueError('Original Family config changed')
    manifest = read_lock(launch['source_manifest'])
    source = root / 'source'; target = source / 'scripts/a2_family_depth_bridge.py'
    for name, digest in manifest['files'].items():
        if sha(source / name) != digest:
            raise ValueError('Original frozen source changed: ' + name)
    for key in ('preparation', 'models_manifest', 'cx_complete', 'mixture_complete',
                'mixture_plan', 'depth_config', 'depth_execution_manifest', 'overlap_preflight'):
        read_lock(cfg[key])
    paused = read_lock(cfg['depth_pause']); read_lock(paused['saved_state'])
    original = target.read_bytes(); gate = Path(project) / 'scripts/a2_family_v14_gate.py'
    if sha(gate) != plan['gate_sha256']:
        raise ValueError('Untested gate code')
    if sha(Path(project) / 'docs/A2_family_spec_v1.4.md') != SPEC_SHA:
        raise ValueError('Wrong preregistration bytes')
    verify_owner(plan)
    transition.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(gate, transition / 'gate.py')
    shutil.copyfile(Path(project) / 'docs/A2_family_spec_v1.4.md', transition / 'spec.md')
    (transition / 'original_depth_bridge.py').write_bytes(original)
    patched = bridge_bytes(original, transition / 'gate.py', sha(gate), transition)
    compile(patched, str(target), 'exec')
    write(transition / 'install-intent.json', dict(plan=plan, original_sha256=sha(target),
        patched_sha256=hashlib.sha256(patched).hexdigest(), state='before_atomic_replace'))
    verify_owner(plan)
    if target.read_bytes() != original: raise ValueError('Bridge changed during installation')
    temporary = target.with_name(target.name + '.v14.tmp')
    with temporary.open('xb') as f:
        f.write(patched); f.flush(); os.fsync(f.fileno())
    os.replace(temporary, target)
    changed = [name for name, digest in manifest['files'].items() if sha(source / name) != digest]
    if changed != ['scripts/a2_family_depth_bridge.py']:
        raise ValueError('Unexpected source changes after hook installation')
    receipt = dict(spec_sha256=SPEC_SHA, original_spec_sha256=cfg['spec_sha256'],
        installed=True, original_manifest=launch['source_manifest'], changed=changed,
        original_sha256=hashlib.sha256(original).hexdigest(), patched_sha256=sha(target),
        gate_sha256=sha(transition / 'gate.py'), family_source_files=len(manifest['files']),
        native_jobs_unchanged=launch['jobs'], transition_root=str(transition),
        ready=False, semantics='Future bridge requires F6/report/backup before original Depth code; no signals, no training modifications',
        ownership=verify_owner(plan))
    write(transition / 'installed.json', receipt)
    print('FAMILY_V14_GATE_INSTALLED', json.dumps(receipt), flush=True)
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--plan', required=True)
    a = p.parse_args(); install(json.loads(Path(a.plan).read_text()), Path.cwd())
