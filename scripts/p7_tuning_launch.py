"""Preflight and narrowly scoped cleanup for the authorized 22-case P7 queue."""
import argparse
import json
from pathlib import Path

from scripts.check_downstream_eval import save
from scripts.p7_tuning_queue import ALL_CASES, make
from scripts.train_then_burn import digest

OLD_NAME = 'q359-p7-exploration8-20261015-a03'
PARTIAL = Path('supervised/run/lm-noaux/seed-42/P7-simple-lm-noaux')
# Inspected W&B link: unlink it without reading/deleting its external target.
EXTERNAL_LOG_LINKS = {
    'wandb/offline-run-20261015_150008-ailzd8af/logs/debug-core.log':
    '/dev/shm/.cache/wandb/logs/core-debug-20261015_150008.log',
}


def read(path):
    return json.loads(Path(path).read_text())


def cleanup_target(source):
    """Refuse completion, wrong identity or symlinks before deleting anything."""
    source = Path(source)
    if source.name != OLD_NAME or source.resolve() != source:
        raise ValueError('Unexpected stopped exploration root')
    stop = read(source / 'user-stop-after-keyonly.json')
    state = read(source / 'supervised/supervisor.json')
    if (stop.get('status') not in ('stop_requested', 'already_terminal')
            or state.get('training_status') != 'failed' or not state.get('finished_at')
            or not state.get('burn', {}).get('collective_progress_verified')
            or stop.get('burn_handoff') != read(source / 'supervised/burn-verified.json')):
        raise ValueError('Old queue is not verified stopped')
    target = source / PARTIAL
    if not target.is_dir() or target.resolve() != target:
        raise ValueError('Interrupted model directory missing or redirected')
    for p in target.rglob('*'):
        if p.is_symlink() and not p.resolve().is_relative_to(target):
            if EXTERNAL_LOG_LINKS.get(str(p.relative_to(target))) != str(p.readlink()):
                raise ValueError('Unrecognized external link in cleanup tree')
    if (target / 'result.json').exists() or (target.parent.parent / 'validated.json').exists():
        raise ValueError('Refuse to delete a completed/result-bearing arm')
    c = read(target / 'train_config.json')
    if (c['pilot']['arm'] != 'P7-simple-lm-noaux' or c['training']['seed'] != 42
            or c['training']['data_seed'] != 42 or c['training']['max_steps'] != 2500
            or c['pilot']['proxy_lambda_max'] != 0):
        raise ValueError('Interrupted arm identity differs')
    for ck in target.glob('checkpoint-*'):
        if int(ck.name.removeprefix('checkpoint-')) >= 2500:
            raise ValueError('Final checkpoint exists; inspect before cleanup')
    return target


def verify_reference(source, run):
    from deep_kv.report import comparable_config
    def normalized(config):
        value = comparable_config(config)
        for key in ('logging_steps','save_steps','eval_steps','eval_on_start'):
            value['training'].pop(key,None)
        for key in ('eval_rows','monitor_rows'):
            value['data'].pop(key,None)
        value.pop('eval_fingerprint',None)
        value['pilot']['proxy_lambda_max'] = .1
        return value
    expected = normalized(read(Path(source)/PARTIAL/'train_config.json'))
    for label,arm,weight,seed in ALL_CASES:
        if seed != 42:
            continue  # Different-seed matching is checked by the aggregate gate.
        actual = read(Path(run)/'smoke'/label/'seed-42'/arm/'train_config.json')
        if normalized(actual) != expected:
            raise ValueError('Smoke recipe/training order differs from previous screen: '+label)


def cleanup(source, run):
    import shutil
    from scripts.gpu_status import require_free
    run = Path(run)
    if read(run / 'smoke-recipes.json')['status'] != 'passed':
        raise ValueError('All numerical/DDP gates must pass before cleanup')
    require_free(list(range(8)))
    target = cleanup_target(source)
    verify_reference(source, run)
    # Preserve a manifest and small diagnostic JSON before the authorized delete.
    files = [p for p in target.rglob('*') if p.is_file() and not p.is_symlink()]
    links = {str(p.relative_to(target)):str(p.readlink()) for p in target.rglob('*') if p.is_symlink()}
    archive = dict(target=str(target), symlinks=links, files=[dict(path=str(p.relative_to(target)), bytes=p.stat().st_size)
                   for p in files], json_files={str(p.relative_to(target)):read(p)
                   for p in files if p.suffix == '.json' and p.stat().st_size < 2**20})
    save(run / 'interrupted-output-before-cleanup.json', archive)
    shutil.rmtree(target)
    if target.exists():
        raise RuntimeError('Interrupted output cleanup incomplete')
    save(run / 'cleanup.json', dict(status='passed', deleted=str(target),
         preserved='Completed arms, queue logs, receipts and shared dataset caches'))


def prepare(root, source, predecessor, rows, host):
    import importlib.metadata as md
    import shutil
    import socket
    import subprocess
    import sys
    import time
    from accelerate.commands.config.config_args import default_yaml_config_file, load_config_from_file
    from scripts.p7_speed_study import predecessor_ready
    from scripts.train_then_burn import approved_launcher, BURN_HASH, GUARD_HASH, burn_progress
    from scripts.verified_gpu_reclaim import inspect, ownership, process
    from scripts.check_fa4_proxy import diagnostic_rows
    from run_experiments import load_jobs
    root, source, predecessor = Path(root), Path(source), Path(predecessor)
    if socket.gethostname() != host or root.exists():
        raise ValueError('Wrong host or output already exists')
    predecessor_ready(predecessor)
    target = cleanup_target(source)
    if (source / 'supervised/run/lm-nexttoken').exists():
        raise ValueError('Next-token production already has output; inspect first')
    expected = dict(torch='2.14.1', transformers='5.9.0', accelerate='1.13.0', datasets='4.8.5',
                    **{'flash-attn-4':'4.0.0b33', 'nvidia-cutlass-dsl':'4.8.0'})
    versions = {k:md.version(k) for k in expected}
    if any(versions[k].split('+')[0] != v for k,v in expected.items()):
        raise ValueError(versions)
    # 220 retained checkpoints plus smoke outputs, with substantial spare space.
    if shutil.disk_usage(root.parent).free < 2500 * 2**30:
        raise ValueError('Need 2500 GiB free for retained checkpoints')
    recipe = read('proxy_heads.12layer.b200.json')
    for key in ('config_name','tokenizer_name','data_dir','eval_data_dir'):
        if not Path(recipe[key]).exists():
            raise ValueError('Missing local ' + key)
    diagnostic_rows(rows)
    root.mkdir()
    cfg_source = Path('resources/accelerate_config.yaml'); destination = Path(default_yaml_config_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.copy2(destination, root / 'accelerate.previous.yaml')
    shutil.copy2(cfg_source, destination)
    ac = load_config_from_file(str(destination)).to_dict()
    if (cfg_source.read_bytes() != destination.read_bytes() or ac['num_processes'] != 8
            or ac['mixed_precision'] != 'bf16' or ac['distributed_type'] != 'MULTI_GPU'):
        raise ValueError('Accelerate configuration differs')
    subprocess.run([str(Path(sys.executable).parent / 'accelerate'), 'env'], check=True)
    jobs = make('proxy_heads.12layer.b200.json', root / 'jobs.json', rows, [c[0] for c in ALL_CASES])
    gate = next(i for i,j in enumerate(jobs) if j['name'] == 'smoke-recipes')
    jobs.insert(gate+1, dict(name='clean-interrupted', argv=[
        '{python}','-m','scripts.p7_tuning_launch','cleanup','--source',str(source),'--root','{run_dir}'],
        required_outputs=[dict(path='cleanup.json',json_equals={'status':'passed'})]))
    (root/'jobs.json').write_text(json.dumps(dict(jobs=jobs),indent=2)+'\n')
    load_jobs(root/'jobs.json')
    status = inspect(); burn = read(predecessor/'supervised/burn-verified.json')
    log = predecessor/'supervised/burn.log'
    if (status['host'] != host or status['guard_disabled'] or len(status['gpus']) != 8
            or any('B200' not in g['name'] or len(g['pids']) != 1 for g in status['gpus'])
            or len(status['workers']) != 8 or ownership(status['gpus']) != ownership(burn['gpus'])
            or digest('/mnt/local/_gpu_guard/gpu_guard.sh') != GUARD_HASH
            or not burn_progress(log.read_text()) or time.time()-log.stat().st_mtime > 120):
        raise ValueError('Current burn ownership/progress differs')
    approved = {str(Path.cwd()/'resources/llm_pretrain_burn.py'): BURN_HASH}
    if any(not approved_launcher(process(pid)['ppid'],approved) for pid in status['workers']):
        raise ValueError('Unexpected GPU workers; no signals sent')
    save(root/'inspection.json',status)
    save(root/'preflight.json',dict(status='passed',versions=versions,environment=sys.executable,
        predecessor=str(predecessor),source=str(source),cleanup_target=str(target),
        cases=ALL_CASES,accelerate_sha256=digest(destination),rows_sha256=digest(rows)))
    print('P7_TUNING_22_READY',flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('prepare','cleanup'))
    p.add_argument('--root',required=True);p.add_argument('--source',required=True)
    p.add_argument('--predecessor');p.add_argument('--rows');p.add_argument('--host')
    a=p.parse_args()
    if a.mode=='cleanup':cleanup(a.source,a.root)
    else:
        if not all((a.predecessor,a.rows,a.host)):p.error('prepare requires predecessor, rows and host')
        prepare(a.root,a.source,a.predecessor,a.rows,a.host)
