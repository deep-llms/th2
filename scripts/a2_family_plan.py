"""Assemble Family inputs and fresh Depth handoff records; never signal a process."""
import argparse
from pathlib import Path
from efficient_llm.a2_family.common import read, write, sha, SPEC_SHA
from scripts.a2_remaining import read_locked
from scripts.a2_quant_io import lock


def inputs(common, mixture, depth, resources, opensci_lock=None):
    common, mixture, depth, resources = map(Path, (common, mixture, depth, resources))
    ready = read(common/'complete.json')
    if ready['status'] != 'complete' or ready['failures']:
        raise ValueError('Common preparation incomplete; leave Depth running')
    model_path = resources/'a2_family_downloads_20261107.json'
    if sha(model_path) != ready['model_manifest_sha256']:
        raise ValueError('Prepared model manifest differs')
    models_ready = read_locked(ready['ready']['models'])
    if models_ready['status'] != 'verified' or models_ready['manifest_sha256'] != sha(model_path):
        raise ValueError('Actual model verification unavailable')
    read_locked(ready['ready']['cx'])
    fwe = read_locked(ready['ready']['fwe'])
    mix_lock = lock(mixture/'complete.json'); mix = read_locked(mix_lock)
    mix_plan = resources/'a2_family_mixture_plan_20261107.json'
    if mix['status'] != 'complete' or mix['plan_sha256'] != sha(mix_plan):
        raise ValueError('Mixture incomplete or source plan differs')
    stream = read_locked(mix['stream'])
    if stream['status'] != 'complete': raise ValueError('Mixture stream incomplete')
    launch_lock = lock(depth/'launch.json'); launch = read_locked(launch_lock)
    depth_cfg = read_locked(launch['config'])
    current = read(depth/'continuations.json')['attempts'][-1]
    if current['config'] != launch['config'] or current['source_manifest'] != launch['source_manifest']:
        raise ValueError('Depth continuation changed original config/source')
    if opensci_lock is None:
        opensci_lock = dict(path='/mnt/local/_outputs/deep-llms_th2/6y3o-a2-inputs-20261028-a01/run/inputs/data/manifest.json', sha256='35aa415f1a3a2a60d9d3faa7faa85e97ce365088f38d23ea01b4a18737834c3e')
    opensci = Path(opensci_lock['path']).parent
    read_locked(opensci_lock)
    flags = [dict(test='mixture_source_qualification', data='mix', detail=f) for f in mix['flags']]
    flags += [dict(test=c['test'], data='mix', detail=c) for c in stream['checks'] if not c['passed']]
    lr = read(resources/'a2_family_lr_state_20261107.json')
    cfg = dict(spec_sha256=SPEC_SHA, preparation=lock(common/'complete.json'),
        models=read(model_path), models_manifest=lock(model_path), cx_root=str(common/'cx'),
        cx_complete=ready['ready']['cx'], streams=fwe['streams'], mixture_complete=mix_lock,
        mixture_stream=dict(path=str(Path(mix['stream']['path']).parent), sha256=mix['stream']['sha256']),
        mixture_plan=lock(mix_plan), input_flags=flags, peak_lr=lr['peak_lr'], lr_state=lr,
        historical=read(resources/'a2_family_historical_20261107.json'),
        k64_reference_nll=2.650000281672266, opensci_cx_root=str(opensci), opensci_manifest=opensci_lock,
        original_depth_source=str(depth/'source'), depth_root=str(depth), depth_launch=launch_lock,
        depth_execution_source=str(Path(current['root'])/'source'),
        depth_execution_manifest=current['execution_source_manifest'],
        depth_execution_root=current['root'], depth_config=launch['config'],
        fwe_pool_documents='/mnt/local/_data/deep-llms_th2/a2_why_fwe_shuffle2030_20261104/documents.bin')
    if set(cfg['streams']) != {'2030','2031'} or not depth_cfg.get('results_root'):
        raise ValueError('Missing registered streams or Depth state')
    return cfg


def pause_request(cfg):
    from scripts.gpu_process import inspect, process
    from scripts.train_then_burn import argv
    from scripts.a2_family_pause_depth import validate_request, verify_owner
    guard = read('/mnt/local/_gpu_guard/DISABLED'); pid = guard['owner_pid']
    live = inspect(); identities = {}; arms = set()
    if len(live['gpus']) != 8 or any(len(g['pids']) != 1 for g in live['gpus']):
        raise ValueError('Expected eight owned training workers')
    for gpu in live['gpus']:
        worker = gpu['pids'][0]; command = argv(worker)
        if '--arm' not in command: raise ValueError('Worker is not a Depth training arm')
        arms.add(command[command.index('--arm')+1]); seen = set()
        while True:
            if worker <= 1 or worker in seen: raise ValueError('Unexpected worker ancestry')
            seen.add(worker); identities[str(worker)] = process(worker)
            if worker == pid: break
            worker = identities[str(worker)]['ppid']
    if len(arms) != 1: raise ValueError('Mixed active Depth arms')
    launch = read_locked(cfg['depth_launch'])
    request = dict(spec_sha256=SPEC_SHA, depth_root=cfg['depth_root'], execution_root=cfg['depth_execution_root'],
        arm=arms.pop(), host=live['host'], supervisor_pid=pid, processes=identities,
        gpu_mapping=[[g['index'],g['uuid'],g['pids']] for g in live['gpus']],
        config=launch['config'], source_manifest=launch['source_manifest'],
        execution_source_manifest=cfg['depth_execution_manifest'],
        authority='User authorized Family v1.3 preemption at next complete checkpoint, then exact Depth resume')
    validate_request(request); verify_owner(request)
    return request


def finalize(cfg, pause):
    from scripts.gpu_process import inspect, process
    from scripts.launch_a2_family import predecessor
    root = Path(cfg['depth_execution_root']); identities = {}
    for gpu in inspect()['gpus']:
        for pid in gpu['pids']:
            worker = process(pid); parent = process(worker['ppid'])
            identities[str(pid)] = worker; identities[str(parent['pid'])] = parent
    cfg = dict(cfg, depth_pause=lock(Path(pause)/'terminal.json'), predecessor=dict(
        root=str(root), supervisor=lock(root/'supervised/supervisor.json'),
        burn=lock(root/'supervised/burn-verified.json'),
        source_manifest=cfg['depth_execution_manifest'],
        burn_script=str(root/'source/resources/llm_pretrain_burn.py'), known_burn_identities=identities))
    predecessor(cfg)
    from scripts.a2_family_depth_bridge import validate
    validate(cfg)
    return cfg


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--mode',choices=['inputs','request','finalize'],required=True)
    for k in ('common','mixture','depth','resources','config','pause'): p.add_argument('--'+k)
    p.add_argument('--output',required=True); a = p.parse_args()
    if Path(a.output).exists(): raise ValueError('Output already exists')
    value = inputs(a.common,a.mixture,a.depth,a.resources) if a.mode=='inputs' else pause_request(read(a.config)) if a.mode=='request' else finalize(read(a.config),a.pause)
    write(a.output,value)
