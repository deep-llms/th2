"""One authorized boundary handoff for the already-running P7 tuning queue.

The old runner has no live queue editing API. Its exclusive log creation is
used as a fail-closed boundary: reserve the NEXT job's log, allowing the current
training and validator to finish. No training PID is signalled. The old
supervisor restores burns before this process starts a fresh remaining queue.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from run_experiments import load_jobs, now, write_json
from scripts.train_then_burn import digest

CURRENT = 'tg-heads2'
NEXT = 'tg-heads8'
PRIORITY = ('tg-valueonly', 'tg-keyonly', 'lm-valueonly-noaux',
            'lm-keyonly-noaux', 'lm-sparse-noaux', 'a-seed2042',
            'lm-noaux-seed2042', 'tg-heads8')


def read(path):
    return json.loads(Path(path).read_text())


def remaining_jobs(jobs):
    start = next((i for i, j in enumerate(jobs) if j['name'] == NEXT+'-train'), None)
    if start is None:
        raise ValueError('Expected next job is absent')
    tail = jobs[start:]
    labels = [j['name'][:-6] for j in tail if j['name'].endswith('-train')]
    if len(labels) != 15 or len(set(labels)) != 15 or not set(PRIORITY) <= set(labels):
        raise ValueError('Expected exactly the authorized fifteen pending arms')
    order = list(PRIORITY) + [label for label in labels if label not in PRIORITY]
    by_name = {j['name']: j for j in tail}
    expected = {label+suffix for label in labels for suffix in ('-train', '-validate')} | {'summary'}
    if set(by_name) != expected:
        raise ValueError('Unexpected pending jobs')
    result = [copy.deepcopy(by_name[label+suffix]) for label in order
              for suffix in ('-train', '-validate')]
    summary = copy.deepcopy(by_name['summary'])
    # Same aggregate gate, limited to the fifteen outputs in the new run root.
    ix = summary['argv'].index('--cases')
    summary['argv'] = summary['argv'][:ix+1] + order
    result.append(summary)
    return order, result


def install(source, root, host):
    if socket.gethostname() != host or root.exists():
        raise ValueError('Wrong host or existing handoff root')
    run = source/'supervised/run'
    jobs = load_jobs(run/'jobs.snapshot.json')
    order, selected = remaining_jobs(jobs)
    state = read(run/'run.json')
    if (state['status'] != 'running' or state['jobs'][-1]['name'] != CURRENT+'-train'
            or state['jobs'][-1]['status'] != 'running'
            or any(j['status'] != 'ok' for j in state['jobs'][:-1])):
        raise ValueError('Current arm changed; inspect before installing boundary')
    if read(run/'smoke-recipes.json')['status'] != 'passed':
        raise ValueError('Original smokes not validated')
    if any((run/label).exists() or (run/(label+'-train.log')).exists() for label in order):
        raise ValueError('A pending arm has already started')
    root.mkdir()
    write_json(root/'jobs.json', {'jobs': selected})
    load_jobs(root/'jobs.json')
    barrier = run/(NEXT+'-train.log')
    request = dict(at=now(), source=str(source), current=CURRENT, blocked=NEXT,
                   priority=list(PRIORITY), order=order,
                   original_manifest_sha256=digest(run/'jobs.snapshot.json'),
                   jobs_sha256=digest(root/'jobs.json'))
    write_json(root/'request.json', request)
    # Exclusive creation: if the old runner got here first, refuse the handoff.
    # Its open('x') happens before Popen, so this cannot kill a running model.
    with barrier.open('x') as f:
        json.dump(dict(kind='authorized_queue_boundary', request=str(root/'request.json')), f)
    write_json(root/'boundary-installed.json', dict(at=now(), path=str(barrier), sha256=digest(barrier)))
    print('BOUNDARY_INSTALLED', json.dumps(request), flush=True)


def boundary_finished(source, root):
    run = source/'supervised/run'
    state = read(run/'run.json')
    supervisor = read(source/'supervised/supervisor.json')
    if state['status'] == 'running' or not supervisor.get('finished_at'):
        return False
    jobs = state['jobs']
    if (state['status'] != 'failed' or jobs[-1]['name'] != NEXT+'-train'
            or jobs[-1]['status'] != 'failed' or 'returncode' in jobs[-1]
            or 'File exists' not in state.get('error', '')
            or any(j['status'] != 'ok' for j in jobs[:-1])
            or jobs[-2]['name'] != CURRENT+'-validate'
            or (run/NEXT).exists()):
        raise ValueError('Old queue did not stop at the requested pre-launch boundary')
    b = read(root/'boundary-installed.json')
    if digest(b['path']) != b['sha256']:
        raise ValueError('Boundary receipt changed')
    if (supervisor.get('training_status') != 'failed' or supervisor.get('handoff_error')
            or not supervisor.get('burn', {}).get('collective_progress_verified')):
        raise ValueError('Old supervisor did not restore verified burns')
    if supervisor['burn'] != read(source/'supervised/burn-verified.json'):
        raise ValueError('Burn handoff receipts differ')
    if Path('/mnt/local/_gpu_guard/DISABLED').exists():
        return False  # Supervisor publishes its receipt just before releasing its guard.
    return True


def launch(source, root, host):
    import shutil
    from accelerate.commands.config.config_args import default_yaml_config_file, load_config_from_file
    from scripts.verified_gpu_reclaim import inspect, ownership, process
    from scripts.train_then_burn import approved_launcher, BURN_HASH, GUARD_HASH, burn_progress
    deadline = time.monotonic()+7200
    while not boundary_finished(source, root):
        if time.monotonic() > deadline:
            raise TimeoutError('No verified queue-boundary handoff within two hours')
        time.sleep(5)
    request = read(root/'request.json')
    if (digest(root/'jobs.json') != request['jobs_sha256'] or
            digest(source/'supervised/run/jobs.snapshot.json') != request['original_manifest_sha256']):
        raise ValueError('Queue manifest changed during wait')
    status = inspect()
    burn = read(source/'supervised/burn-verified.json')
    log = source/'supervised/burn.log'
    if (socket.gethostname() != host or status['host'] != host or status['guard_disabled']
            or len(status['gpus']) != 8 or len(status['workers']) != 8
            or any('B200' not in g['name'] or len(g['pids']) != 1 for g in status['gpus'])
            or ownership(status['gpus']) != ownership(burn['gpus'])
            or digest('/mnt/local/_gpu_guard/gpu_guard.sh') != GUARD_HASH
            or not burn_progress(log.read_text()) or time.time()-log.stat().st_mtime > 120):
        raise ValueError('Live burn verification failed; GPUs untouched')
    approved = {str(Path.cwd()/'resources/llm_pretrain_burn.py'): BURN_HASH}
    if any(not approved_launcher(process(pid)['ppid'], approved) for pid in status['workers']):
        raise ValueError('Unrecognized GPU process; GPUs untouched')
    cfg = Path('resources/accelerate_config.yaml')
    destination = Path(default_yaml_config_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.copy2(destination, root/'accelerate.previous.yaml')
    shutil.copy2(cfg, destination)
    ac = load_config_from_file(str(destination)).to_dict()
    if (cfg.read_bytes() != destination.read_bytes() or ac['num_processes'] != 8
            or ac['mixed_precision'] != 'bf16' or ac['distributed_type'] != 'MULTI_GPU'):
        raise ValueError('Accelerate config mismatch')
    subprocess.run([str(Path(sys.executable).parent/'accelerate'), 'env'], check=True)
    write_json(root/'inspection.json', status)
    write_json(root/'handoff-verified.json', dict(at=now(), burn=burn, order=request['order'],
                                               accelerate_sha256=digest(destination)))
    print('HANDOFF_VERIFIED', now(), flush=True)
    os.execv(sys.executable, [sys.executable, '-u', '-m', 'scripts.train_then_burn',
             '--config', str(root/'jobs.json'), '--output', str(root/'supervised'),
             '--inspection', str(root/'inspection.json'), '--host', host,
             '--burn-session', root.name+'-final-burn'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--host', required=True)
    a = p.parse_args()
    install(a.source, a.root, a.host)
    launch(a.source, a.root, a.host)
