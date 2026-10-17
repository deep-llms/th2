"""Five authorized, checkpoint-based follow-ups; external labels stay local.

The existing supervisor owns reclaim, cancellation and automatic burn restoration.
One fit per physical GPU; smoke checkpoints are never scientific parents.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from deadline_study import training_command, run_child, save, validate_checkpoint

SRC = Path(__file__).resolve().parent
PARENT_ROOT = Path('/mnt/local/_outputs/deep-llms_th2/q359-ce7454-scratch-20261016-a04/supervised/run/study')
EXTERNAL = Path('/mnt/local/_data/deep-llms_th2/ce7454/evaluation/cache')
FINISH_BY = '2026-10-17T04:30:00+00:00'
PARENTS = {
    'grip_nodown': ('grip_nodown_base', 450, 'cd87bc0dbc5cc884203c9cfba1d97e5fd4992923729da9e8555f0ac5e4cb88d1'),
    'grip_nodown_full512': ('grip_nodown_full512', 40, '29090be74fddeb5a77e43795d8c798aef61f26332ff971883631e829dc58fb8c'),
    'srresnet': ('srresnet_full512', 40, '46da5542ff9e7e9ff103db9cb8e4e8e3e7b24ed1506d778d62495b86b43b0571'),
    'srresnet_stats': ('srresnet_stats_full512', 40, 'e16f3c6c23b58e49cd7a4335144ca18a81faf203acbeba98a9a189f9bdfe93b6'),
}
ARMS = [dict(name=f'{model}_s{seed}', model=model, parent=parent, lr=lr,
             seed=seed, data_seed=42+seed, gpu=gpu)
        for gpu, (model, parent, lr, seed) in enumerate(
            (model, parent, lr, seed)
            for model, parent, lr in [('grip_nodown_ibna', 'grip_nodown', 1e-4),
                                     ('grip_nodown_ibnb', 'grip_nodown', 1e-4),
                                     ('grip_nodown', 'grip_nodown_full512', 3e-5),
                                     ('srresnet', 'srresnet', 3e-5),
                                     ('srresnet_stats', 'srresnet_stats', 3e-5)]
            for seed in (1,))]


def parent_path(arm):
    return PARENT_ROOT/PARENTS[arm['parent']][0]/'last.pt'


def command(arm, cache, output, epochs=40):
    cmd = training_command(arm['model'], cache, output, epochs,
                           final=True, parent=parent_path(arm))
    for flag, value in [('--lr', arm['lr']), ('--seed', arm['seed']),
                        ('--data-seed', arm['data_seed'])]:
        cmd[cmd.index(flag)+1] = str(value)
    return cmd


def verify_parents():
    import torch
    torch.set_num_threads(1)
    receipts = {}
    for model, (folder, epoch, expected) in PARENTS.items():
        path = PARENT_ROOT/folder/'last.pt'
        with path.open('rb') as f:
            actual = hashlib.file_digest(f, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError(f'Parent changed: {path}')
        ck = torch.load(path, map_location='cpu', weights_only=False)
        a = ck['args']
        if ck['epoch'] != epoch or any(a.get(k) != v for k, v in
                {'model': folder.removesuffix('_base').removesuffix('_full512'),
                 'train_split': 'train', 'split_seed': 42,
                 'val_frac': .1, 'seed': 0, 'data_seed': 42, 'kd_teacher': None}.items()):
            raise ValueError(f'Wrong parent recipe: {path}')
        receipts[model] = {'path': str(path), 'sha256': actual, 'epoch': epoch}
    return receipts


def validate_fit(arm, output, epochs):
    import torch
    torch.set_num_threads(1)
    path = output/'last.pt'
    result = validate_checkpoint(path, arm['model'], epochs, parent_path(arm),
                                 seed=arm['seed'], data_seed=arm['data_seed'])
    ck = torch.load(path, map_location='cpu', weights_only=False)
    expected = {'crop': 512, 'rrc_size': 512, 'batch_size': 16, 'lr': arm['lr'],
                'optim': 'adam', 'sched': 'cosine', 'warmup_epochs': 0}
    if any(ck['args'].get(k) != v for k, v in expected.items()):
        raise ValueError('Unexpected continuation recipe')
    with (output/'log.csv').open() as f:
        rows = list(csv.DictReader(f))
    if len(rows) != epochs or [int(r['epoch']) for r in rows] != list(range(1, epochs+1)):
        raise ValueError('Incomplete epoch history')
    if any(not math.isfinite(float(v)) for r in rows for v in r.values()):
        raise ValueError('Nonfinite training history')
    result['last_epoch_seconds'] = float(rows[-1]['sec'])
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preflight-only', action='store_true')
    p.add_argument('--cache', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--finish-by', default=FINISH_BY)
    args = p.parse_args()
    parents = verify_parents()
    ready = json.loads((EXTERNAL/'ready.json').read_text())
    if ready['status'] != 'passed' or ready['purpose'] != 'evaluation_only':
        raise ValueError('Unverified external evaluation cache')
    if args.preflight_only:
        print('FOLLOWUP_PARENTS_VERIFIED', json.dumps(parents), flush=True)
        return
    if not args.cache or not args.output or os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1,2,3,4,5,6,7':
        raise ValueError('Expected cache, fresh output and supervisor allocation of all eight GPUs')
    deadline = datetime.fromisoformat(args.finish_by).timestamp()
    if deadline-time.time() < 3*3600:
        raise RuntimeError('Insufficient time for smoke, training and evaluation')
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output/'manifest.json', {'arms': ARMS, 'parents': parents, 'epochs': 40,
         'resolution': 512, 'finish_by': args.finish_by,
         'note': 'Only own scratch-trained parents; fresh optimizers, not exact optimizer resume. Fixed train split. No external labels.'})

    def smoke(arm):
        name, gpu = arm['name'], arm['gpu']
        # New normalization architectures receive the established numerical gate.
        if arm['model'] in ('grip_nodown_ibna', 'grip_nodown_ibnb'):
            dest = args.output/('numerics_'+name)
            run_child([sys.executable, '-u', str(SRC/'benchmark_deadline_candidates.py'),
                       '--model', arm['model'], '--cache', str(args.cache),
                       '--output', str(dest), '--crop', '512', '--batch-size', '16'],
                      gpu, args.output/(name+'_numerics.log'), deadline)
            r = json.loads((dest/'result.json').read_text())
            if r['steps'] != 30 or not r['full512_reload_exact'] or r['parameters'] > 12300000:
                raise ValueError('Failed normalization numerical gate')
        dest = args.output/('entrypoint_'+name)
        run_child(command(arm, args.cache, dest, 1), gpu,
                  args.output/(name+'_entrypoint.log'), deadline)
        return validate_fit(arm, dest, 1)

    with ThreadPoolExecutor(max_workers=len(ARMS)) as pool:
        smokes = list(pool.map(smoke, ARMS))
    estimate = max(r['last_epoch_seconds'] for r in smokes)*40*1.5+1200
    if estimate >= deadline-time.time():
        raise RuntimeError('Measured continuation cannot safely fit deadline')
    save(args.output/'smoke_passed.json', {'status': 'passed', 'results': smokes,
         'estimated_seconds_with_margin': estimate})
    print('FOLLOWUP_SMOKE_PASSED', estimate, flush=True)

    def fit(arm):
        name, gpu = arm['name'], arm['gpu']
        dest = args.output/name
        run_child(command(arm, args.cache, dest), gpu, args.output/(name+'.log'), deadline)
        checkpoint = validate_fit(arm, dest, 40)
        evaluation = args.output/(name+'_evaluation')
        run_child([sys.executable, '-u', str(SRC/'evaluate_deadline.py'),
                   '--checkpoint', str(dest/'last.pt'), '--cache', str(args.cache),
                   '--external-cache', str(EXTERNAL), '--output', str(evaluation)],
                  gpu, args.output/(name+'_evaluation.log'), deadline)
        result = json.loads((evaluation/'results.json').read_text())
        if result['status'] != 'passed' or result['checkpoint_sha256'] != checkpoint['sha256']:
            raise ValueError('Final evaluation identity mismatch')
        external = result['external_predictions']
        if external['status'] != 'passed' or any(external['splits'][s]['n'] != n
                for s, n in [('dev', 900), ('test_official', 1500)]):
            raise ValueError('Incomplete external predictions')
        return {'arm': name, 'checkpoint': checkpoint, 'evaluation': result}

    with ThreadPoolExecutor(max_workers=len(ARMS)) as pool:
        results = list(pool.map(fit, ARMS))
    if verify_parents() != parents:
        raise ValueError('Parent checkpoint changed during follow-up')
    save(args.output/'complete.json', {'status': 'passed', 'results': results,
         'finished_utc': datetime.now(timezone.utc).isoformat()})


if __name__ == '__main__':
    main()
