"""Bind continuation to its actual failed predecessor and known burn ownership."""
from pathlib import Path
from scripts.a2_remaining import read_locked
from scripts.train_then_burn import digest,argv
from scripts.gpu_process import inspect,process


def verify_predecessor(bundle_lock,output_parent,burn_script):
    b=read_locked(bundle_lock);root=Path(b['root']);policy=read_locked(b['policy'])
    if (root.parent.resolve()!=Path(output_parent).resolve()
            or Path(burn_script).resolve()!=root/'source/resources/llm_pretrain_burn.py'
            or policy['production_batch_size']!=2 or policy['full_set_LL']!='diagnostic_only'):
        raise ValueError('Unrecognized continuation scope or predecessor')
    sv=read_locked(b['supervisor']);burn=read_locked(b['burn']);manifest=read_locked(b['source_manifest'])
    if (sv['training_status']!='failed' or not sv.get('finished_at')
            or len(sv['free_after_training'])!=8 or any(g['pids'] for g in sv['free_after_training'])
            or burn['collective_progress_verified'] is not True):
        raise ValueError('Predecessor has not released its workload')
    if manifest['code_commit']!=b['source_commit']:raise ValueError('Wrong predecessor source')
    for rel,expected in manifest['files'].items():
        if digest(root/'source'/rel)!=expected:raise ValueError('Frozen predecessor source changed')
    prior_plan=read_locked(b['plan']);decision=read_locked(b['decision'])
    for name in ('scores','replays','predictions'):
        for item in decision['inputs'][name].values():read_locked(item)
    if decision['inputs']['prepared']['sha256']!=prior_plan['prepared']['sha256']:
        raise ValueError('Canonical request identity changed')
    # No stale-PID inference: inspect actual workers, exact command and start identity
    # now; the native supervisor rechecks via pidfds immediately before signaling.
    current=inspect()
    expected={(g['index'],g['uuid'],tuple(g['pids'])) for g in burn['gpus']}
    actual={(g['index'],g['uuid'],tuple(g['pids'])) for g in current['gpus']}
    if expected!=actual:raise ValueError('Known predecessor GPU workers changed; no signals sent')
    records={}
    for gpu in current['gpus']:
        for pid in gpu['pids']:
            worker=process(pid);launcher=process(worker['ppid'])
            if str(burn_script) not in argv(worker['ppid']):raise ValueError('Different burn ownership')
            records[str(pid)]=worker;records[str(worker['ppid'])]=launcher
    if records!=b['known_burn_identities']:raise ValueError('Known burn start/command identities changed')
    return dict(bundle=bundle_lock,inspection=current,identities=records)
