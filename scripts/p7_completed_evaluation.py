"""Evaluate completed exploration arms after an explicitly authorized queue stop.

Reuses the existing nine-task protocol and eight-GPU evaluation dispatcher.
This preflight does not signal processes or launch work.
"""
import argparse
import json
from pathlib import Path

from scripts.p7_exploration_queue import CASES, checkpoint_path, validate_training
from scripts.proxy_confirmation_queue import evaluation_reference, preflight
from scripts.proxy_screen_evaluation import make_jobs
from scripts.check_downstream_eval import save
from scripts.train_then_burn import digest


def read(path):
    return json.loads(Path(path).read_text())


def selected_cases(training_root):
    root = Path(training_root)
    cases = tuple(case for case in CASES
                  if (root / case[0] / 'validated.json').is_file()
                  and (checkpoint_path(root, case).parent / 'result.json').is_file())
    if cases != CASES[:len(cases)] or len(cases) < 5:
        raise ValueError('Require a completed prefix through key-only')
    return cases


def identities(scope):
    labels = scope['completed_labels']
    cases = tuple(case for case in CASES if case[0] in labels)
    if len(cases) < 5 or labels != [c[0] for c in CASES[:len(cases)]]:
        raise ValueError('Changed completed exploration selection')
    items = [(str(checkpoint_path(scope['training_root'], c)), *c) for c in cases]
    for label, arm in (('short-A', 'A'), ('short-lm', 'P7-simple-lm')):
        ck = Path(scope['reference_screen']) / 'supervised/run/seed-42' / arm / 'checkpoint-2500'
        items.append((str(ck), label, arm, 2500, .1))
    return items


def validate_scope(scope):
    items = identities(scope)
    if scope['checkpoints'] != [row[0] for row in items] or set(scope['specs']) != set(scope['checkpoints']):
        raise ValueError('Changed checkpoint paths/order')
    for ck, label, arm, steps, weight in items:
        spec = scope['specs'][ck]
        if (spec['label'], spec['arm'], spec['step'], spec['lambda_max'], spec['seed']) != (label, arm, steps, weight, 42):
            raise ValueError('Changed checkpoint identity')


def build_scope(training_root, reference):
    from deep_kv.report import report
    previous = evaluation_reference(reference)
    cases = selected_cases(training_root)
    rows = validate_training(training_root, cases=cases)
    scope = dict(kind='p7-completed', training_root=str(training_root),
                 completed_labels=[c[0] for c in cases], reference_screen=previous['source_screen'],
                 reference=str(reference), protocols=previous['protocols'],
                 data_root=previous['data_root'], manifest=previous['manifest'],
                 manifest_sha256=previous['manifest_sha256'], versions=previous['versions'],
                 evaluation_seed=42, finetuning=False, paired_loss=False)
    specs = {}; fingerprints = None; tokenizer = None
    for ck, label, arm, steps, weight in identities(scope):
        folder = Path(ck).parent
        c = read(folder / 'train_config.json'); r = read(folder / 'result.json')
        report(folder.parent, [arm], expected_step=steps, expected_seed=42, expected_attention_backend='fa4')
        if (r['status'] != 'complete' or r['global_step'] != steps
                or c['training']['max_steps'] != steps or c['training']['warmup_steps'] != steps // 20
                or c['training']['seed'] != 42 or c['training']['data_seed'] != 42
                or c['model_config']['num_hidden_layers'] != 12 or not c['training']['bf16']
                or c['pilot']['proxy_lambda_max'] != weight
                or read(Path(ck) / 'trainer_state.json')['global_step'] != steps):
            raise ValueError('Invalid completed checkpoint: ' + label)
        pair = (c['train_fingerprint'], c['eval_fingerprint'])
        if fingerprints is None:
            fingerprints, tokenizer = pair, c['model']['tokenizer_name']
        if pair != fingerprints or tokenizer != c['model']['tokenizer_name']:
            raise ValueError('Checkpoint data/tokenizer mismatch')
        specs[ck] = dict(label=label, arm=arm, step=steps, lambda_max=weight, seed=42,
                         checkpoint_sha256=digest(Path(ck) / 'model.safetensors'))
    scope.update(checkpoints=list(specs), specs=specs)
    validate_scope(scope)
    return scope, rows


def retry_burn(predecessor, scope):
    predecessor = Path(predecessor)
    previous = read(predecessor / 'supervised/supervisor.json')
    run = read(predecessor / 'supervised/run/run.json')
    burn = read(predecessor / 'supervised/burn-verified.json')
    if (read(predecessor / 'scope.json') != scope or previous.get('training_status') != 'failed'
            or run.get('status') != 'failed' or not previous.get('finished_at')
            or not previous.get('burn', {}).get('collective_progress_verified')
            or not burn.get('collective_progress_verified')
            or burn['session'] != predecessor.name + '-final-burn'
            or previous['burn']['session'] != burn['session']):
        raise ValueError('Retry predecessor scope/final burn handoff differs')
    return burn


def prepare(root, source, reference, host, predecessor=None, *, tuning_sources=None,
            stop_receipt='user-stop-after-keyonly.json'):
    import shutil
    import socket
    import subprocess
    import sys
    import time
    from accelerate.commands.config.config_args import default_yaml_config_file, load_config_from_file
    from scripts.train_then_burn import approved_launcher, BURN_HASH, GUARD_HASH, burn_progress
    from scripts.verified_gpu_reclaim import inspect, ownership, process
    from run_experiments import load_jobs
    root, source = Path(root), Path(source)
    if socket.gethostname() != host or root.exists():
        raise ValueError('Wrong host or output already exists')
    stop = read(source / stop_receipt)
    state = read(source / 'supervised/supervisor.json')
    burn = read(source / 'supervised/burn-verified.json')
    if (stop['status'] not in ('stop_requested', 'already_terminal') or not state.get('finished_at')
            or not state.get('burn', {}).get('collective_progress_verified')
            or not burn.get('collective_progress_verified')
            or burn['session'] != source.name + '-final-burn'
            or state['burn']['session'] != burn['session']
            or stop['burn_handoff'] != burn):
        raise ValueError('Authorized stop/burn handoff is not verified')
    # Existing benchmark and environment preflight is reused unchanged.
    root.mkdir(parents=True)
    preflight(root, reference)
    if tuning_sources is None:
        scope, rows = build_scope(source / 'supervised/run', reference)
    else:
        from scripts.p7_tuning_evaluation import build_scope as build_tuning_scope
        scope, rows = build_tuning_scope(tuning_sources, reference)
    burn_source = source
    if predecessor is not None:
        burn_source = Path(predecessor)
        burn = retry_burn(burn_source, scope)
    config_source = Path('resources/accelerate_config.yaml'); destination = Path(default_yaml_config_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.copy2(destination, root / 'accelerate.previous.yaml')
    shutil.copy2(config_source, destination)
    ac = load_config_from_file(str(destination)).to_dict()
    if (config_source.read_bytes() != destination.read_bytes() or ac['num_processes'] != 8
            or ac['mixed_precision'] != 'bf16' or ac['distributed_type'] != 'MULTI_GPU'):
        raise ValueError('Accelerate configuration differs')
    subprocess.run([str(Path(sys.executable).parent / 'accelerate'), 'env'], check=True)
    status = inspect()
    log = burn_source / 'supervised/burn.log'
    if (status['host'] != host or status['guard_disabled'] or len(status['gpus']) != 8
            or any('B200' not in g['name'] or len(g['pids']) != 1 for g in status['gpus'])
            or len(status['workers']) != 8 or ownership(status['gpus']) != ownership(burn['gpus'])
            or digest('/mnt/local/_gpu_guard/gpu_guard.sh') != GUARD_HASH
            or not burn_progress(log.read_text()) or time.time() - log.stat().st_mtime > 120):
        raise ValueError('Current burn ownership/progress differs')
    approved = {str(Path.cwd() / 'resources/llm_pretrain_burn.py'): BURN_HASH}
    if any(not approved_launcher(process(pid)['ppid'], approved) for pid in status['workers']):
        raise ValueError('Unexpected GPU worker; no signals sent')
    save(root / 'scope.json', scope)
    save(root / 'training-results.json', rows)
    save(root / 'jobs.json', dict(jobs=make_jobs(root / 'scope.json')))
    load_jobs(root / 'jobs.json')
    save(root / 'inspection.json', status)
    save(root / 'preflight.json', dict(status='passed', checkpoints=scope['checkpoints'],
         accelerate_sha256=digest(destination), source=str(source), reference=str(reference),
         retry_predecessor=str(predecessor) if predecessor is not None else None))
    print('COMPLETED_EXPLORATION_EVALUATION_READY', scope['completed_labels'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True); p.add_argument('--source', required=True)
    p.add_argument('--reference', required=True); p.add_argument('--host', required=True)
    p.add_argument('--predecessor', help='Explicit failed evaluation root for a fresh retry')
    p.add_argument('--tuning-sources', nargs='+', help='Completed tuning run directories, frozen before evaluation')
    p.add_argument('--stop-receipt', default='user-stop-after-keyonly.json')
    a = p.parse_args()
    prepare(a.root, a.source, a.reference, a.host, a.predecessor,
            tuning_sources=a.tuning_sources, stop_receipt=a.stop_receipt)
