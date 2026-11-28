"""L34 on the B200 pq5r (28 Nov 2026): verified native-burn handoff, one-GPU DOT / Huginn fits, downstream scores,
native burn restored.

Pattern of scripts/anchor_long_queue.supervise (used for the OHMV rounds), parametrized for a fresh node:
  1. node, GPU UUIDs and every burn worker / launcher identity must equal the read-only inspection receipt
     (pinned by SHA256), and each launcher must run the inspected native script with its inspected hash;
  2. the GPU guard (if the node has one) is disabled by an exclusive marker that only this supervisor removes;
  3. scripts.train_then_burn.stop_burns signals only those verified worker PIDs (pidfd SIGKILL, never a process
     group, never a name pattern), waits 30 s and requires all eight GPUs free;
  4. one `scripts/local_screen.py run` per job on its own GPU (the tested local trainer; rows hash-pinned), then
     `scripts/local_downstream.py score` for every finished fit;
  5. finally (also on failure): only this supervisor's own descendants are stopped, the GPUs must be free, the
     inspected native burn is relaunched in a new tmux session and its eight workers verified twice.
  python scripts/node_dot_queue.py --base <new dir> --rows <rows dir> --requests <requests.json> \
      --inspection <receipt> --inspection-sha256 <sha> --jobs 0:arm 1:arm ...
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from efficient_llm.warm_order.data import sha  # noqa: E402

GUARD = Path('/mnt/local/_gpu_guard')
GPUS = list(range(8))


def write_json(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=1, default=str) + '\n')
    os.replace(tmp, path)


def native_of(receipt):
    """The inspected native burn: (script path, sha256), required identical for all eight launchers."""
    scripts = set()
    for pid in receipt['workers']:
        chain = receipt['ancestor_chains'][str(pid)]
        launcher = chain[1] if len(chain) > 1 else None
        argv = [a for a in (launcher or {}).get('argv', [])[1:] if a not in ('-u', '-B')]
        if not argv or not argv[0].endswith('.py'):
            raise RuntimeError(f'worker {pid}: launcher is not a python script invocation')
        scripts.add(argv[0])
    if len(scripts) != 1:
        raise RuntimeError(f'expected one native burn script, found {sorted(scripts)}')
    script = scripts.pop()
    digest = receipt['files'].get(script, {}).get('sha256')
    if not digest or sha(script) != digest:
        raise RuntimeError('native burn script missing from the receipt or changed since inspection')
    return script, digest


def check_ownership(receipt, inspect, process, ownership, approved_launcher, native):
    """Current GPU workers must be exactly the inspected ones, with unchanged identities and approved launchers."""
    now = inspect()
    if ownership(now['gpus']) != ownership(receipt['gpus']) or len(now['workers']) != 8:
        raise RuntimeError('GPU ownership changed since inspection')
    for pid, record in receipt['processes'].items():
        if process(int(pid)) != record:
            raise RuntimeError(f'process {pid} identity changed since inspection')
    if any(not approved_launcher(process(pid)['ppid'], {native[0]: native[1]}) for pid in now['workers']):
        raise RuntimeError('a GPU worker is not owned by the inspected native burn')
    return now


def run_jobs(jobs, base, rows, requests, python=sys.executable, poll=20, gpus=None):
    """jobs: [(seed, arm)], any number, queued over the GPUs. Each GPU runs one job at a time: its fit, then (on fit
    success) its downstream score on the same GPU, then the next pending job, so no GPU idles while work remains."""
    gpus = list(GPUS if gpus is None else gpus)
    (base/'logs').mkdir(parents=True, exist_ok=True)
    pending, results, running = list(jobs), {}, {}

    def start(gpu, seed, arm, stage):
        argv = ([python, '-u', 'scripts/local_screen.py', 'run', '--rows', str(rows), '--base', str(base),
                 '--seed', str(seed), '--arm', arm] if stage == 'fit' else
                [python, '-u', 'scripts/local_downstream.py', 'score', '--base', str(base), '--seed', str(seed),
                 '--arm', arm, '--requests', str(requests)])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='8', PYTHONDONTWRITEBYTECODE='1')
        log = (base/'logs'/f'{stage}-s{seed}-{arm}.log').open('x')
        running[gpu] = (subprocess.Popen(argv, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT),
                        seed, arm, stage)
        log.close()
        print(time.strftime('%H:%M:%S'), 'START', stage, gpu, f's{seed}/{arm}', flush=True)

    while pending or running:
        for gpu in gpus:
            if gpu in running:
                child, seed, arm, stage = running[gpu]
                if child.poll() is None:
                    continue
                key = f's{seed}/{arm}'
                results.setdefault(key, {})[stage] = child.returncode
                print(time.strftime('%H:%M:%S'), 'END', stage, key, 'rc', child.returncode, flush=True)
                del running[gpu]
                if stage == 'fit' and child.returncode == 0:
                    start(gpu, seed, arm, 'downstream')
                    continue
            if gpu not in running and pending:
                seed, arm = pending.pop(0)
                start(gpu, seed, arm, 'fit')
        write_json(base/'jobs.json', dict(results=results, pending=[f's{s}/{a}' for s, a in pending],
                                          running={str(g): f's{r[1]}/{r[2]} {r[3]}' for g, r in running.items()}))
        if pending or running:
            time.sleep(poll)
    return results


def restore_native(base, native, session, inspect, process, approved_launcher, require_free):
    """Relaunch the inspected native burn exactly as inspected (system python, all eight GPUs), verify twice."""
    script, digest = native
    require_free(GPUS)
    if sha(script) != digest:
        raise RuntimeError('native script changed')
    if subprocess.run(['tmux', 'has-session', '-t', session], capture_output=True).returncode == 0:
        raise RuntimeError('native session already exists')
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    command = 'exec ' + shlex.join(['env', 'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7', 'MASTER_ADDR=127.0.0.1',
                                    f'MASTER_PORT={port}', '/usr/bin/python3', script])
    command += ' >' + shlex.quote(str(base/'native.log')) + ' 2>&1'
    subprocess.run(['tmux', 'new-session', '-d', '-s', session, command], check=True)
    subprocess.run(['tmux', 'set-option', '-w', '-t', session, 'remain-on-exit', 'on'], check=True)
    samples = []
    for _ in range(2):
        time.sleep(30)
        state = inspect()
        if len(state['workers']) != 8 or any(len(g['pids']) != 1 for g in state['gpus']):
            raise RuntimeError('native GPU mapping incomplete')
        if any(not approved_launcher(process(pid)['ppid'], {script: digest}) for pid in state['workers']):
            raise RuntimeError('native owner mismatch')
        pane = int(subprocess.check_output(['tmux', 'display-message', '-p', '-t', session, '#{pane_pid}'], text=True))
        if {process(pid)['ppid'] for pid in state['workers']} != {pane}:
            raise RuntimeError('native launcher is not the owned tmux pane')
        samples.append(state)
    write_json(base/'native-verified.json', dict(samples=samples, script=script, script_sha256=digest))


def native_session(base):
    """tmux session for the restored native burn: unique per batch (an earlier batch's restored burn keeps its own)."""
    return f'{base.parent.name}_{base.name}_native'.replace('.', '_')


def supervise(a):
    from scripts.gpu_process import inspect, ownership, process
    from scripts.gpu_status import require_free
    from scripts.train_then_burn import GUARD_HASH, approved_launcher, clean_owned_children, enable_subreaper, \
        stop_burns
    if sha(a.inspection) != a.inspection_sha256:
        raise RuntimeError('inspection receipt mismatch')
    receipt = json.loads(a.inspection.read_text())
    if socket.gethostname() != receipt['host']:
        raise RuntimeError('node changed since inspection')
    jobs = [(int(j.split(':', 1)[0]), j.split(':', 1)[1]) for j in a.jobs]
    from efficient_llm.warm_order.loop_screen import SPECS
    if any(arm not in SPECS for _, arm in jobs) or len(set(jobs)) != len(jobs):
        raise ValueError('unknown or duplicate job')
    if sha(a.requests) != '6282100c39545c6d14b219a4a6538e4ecf13b94a12b97c19c68c18c61a450091':
        raise RuntimeError('downstream requests differ from the pin')
    pins = json.loads((ROOT/'docs/results/local_rows_20261126/manifest.json').read_text())['files']
    if json.loads((a.rows/'manifest.json').read_text())['files'] != pins:
        raise RuntimeError('row pins differ from the local pins')
    native = native_of(receipt)
    from scripts.train_then_burn import APPROVED_BURNS
    if APPROVED_BURNS.get(native[0]) != native[1]:  # defense in depth: only a reviewed native burn is ever stopped
        raise RuntimeError(f'native burn {native} is not a reviewed script/hash')
    session = native_session(a.base)
    if subprocess.run(['tmux', 'has-session', '-t', session], capture_output=True).returncode == 0:
        raise RuntimeError(f'native session {session} already exists; refusing before any stop')
    a.base.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(a.base).free < 200 * 1024**3:
        raise RuntimeError('insufficient disk')
    write_json(a.base/'admission.json', dict(jobs=jobs, native=native, inspection_sha256=a.inspection_sha256,
                                             utc=time.time()))
    guard_marker = None
    if GUARD.is_dir() and (GUARD/'gpu_guard.sh').is_file():
        if sha(GUARD/'gpu_guard.sh') != GUARD_HASH:
            raise RuntimeError('guard script differs from the reviewed version')
        guard_marker = json.dumps(dict(owner_pid=os.getpid(), base=str(a.base), utc=time.time()))
        with (GUARD/'DISABLED').open('x') as f:
            f.write(guard_marker)
    enable_subreaper()

    def interrupt(*_):
        raise KeyboardInterrupt('supervisor interrupted')
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    reclaimed = success = False
    results = {}
    try:
        time.sleep(30)  # let any guard pass already in flight finish
        before = check_ownership(receipt, inspect, process, ownership, approved_launcher, native)
        write_json(a.base/'before-reclaim.json', before)
        reclaimed = True
        stop_burns(a.base, {native[0]: native[1]})
        results = run_jobs(jobs, a.base, a.rows, a.requests)
        success = all(r.get('fit') == 0 and r.get('downstream') == 0 for r in results.values()) and \
            len(results) == len(jobs)
    except BaseException as failure:
        write_json(a.base/'failure.json', dict(reason=repr(failure), utc=time.time()))
        traceback.print_exc()
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        restored = False
        try:
            if reclaimed:
                clean_owned_children()
                time.sleep(30)
                restore_native(a.base, native, session, inspect, process, approved_launcher, require_free)
                restored = True
        except BaseException as failure:
            write_json(a.base/'handoff-failure.json', dict(reason=repr(failure), utc=time.time()))
            traceback.print_exc()
        if guard_marker is not None and (restored or not reclaimed):  # never re-enable the guard on uncertain ownership
            if (GUARD/'DISABLED').read_text() == guard_marker:
                (GUARD/'DISABLED').unlink()
        write_json(a.base/'terminal.json', dict(success=success, native_restored=restored, results=results,
                                                utc=time.time()))
        print('L34_TERMINAL', json.dumps(dict(success=success, native_restored=restored)), flush=True)
    return 0 if success else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--rows', type=Path, required=True)
    p.add_argument('--requests', type=Path, required=True)
    p.add_argument('--inspection', type=Path, required=True)
    p.add_argument('--inspection-sha256', required=True)
    p.add_argument('--jobs', nargs='+', required=True)
    sys.exit(supervise(p.parse_args()))
