"""Four independent single-GPU fits, with a shared budget fixed from timings.

Run beneath the existing train_then_burn supervisor; never reclaim GPUs here.
All child processes stay in the supervisor-owned process session.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ARMS = ['grip_nodown', 'srresnet', 'srresnet_stats', 'grip_nodown_stats']
SRC = Path(__file__).resolve().parent
FINISH_BY = '2026-10-17T02:00:00+00:00'


def save(path, data):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def training_command(model, cache, output, epochs, *, final=False, parent=None):
    cmd = [sys.executable, '-u', str(SRC / 'train.py'), '--model', model,
           '--cache-dir', str(cache), '--out-root', str(output.parent),
           '--run-name', output.name, '--epochs', str(epochs),
           '--warmup-epochs', '0', '--batch-size', '16' if final else '32',
           '--val-batch-size', '32', '--lr', '0.0001' if final else '0.0003',
           '--weight-decay', '0.0001', '--label-smoothing', '0', '--clip-grad', '0',
           '--crop', '512' if final else '256', '--rrc-size', '512' if final else '256',
           '--val-frac', '0.1', '--split-seed', '42', '--seed', '0', '--data-seed', '42',
           '--workers', '8', '--norm', 'imagenet', '--optim', 'adam', '--sched', 'cosine',
           '--deterministic', '--flips', 'rot90_hflip', '--views', '2', '--jsd-lambda', '0',
           '--grip-blur-p', '0.1', '--grip-blur-smax', '3', '--rrc-p', '0.2',
           '--sharpen-p', '0.5', '--sharpen-mode', 'composite',
           '--social-mode', 'cf', '--social-p', '1', '--social-region', '--social-cf-interp']
    if final:
        if parent is None:
            raise ValueError('Final stage requires a verified parent')
        cmd += ['--crop-upscale', '--init-from', str(parent)]
    return cmd


def select_budget(reports, available_seconds):
    # Includes loader wait and full512 validation every epoch, plus conservative
    # margin for shared CPU/storage contention and checkpoint writing.
    costs = {}
    for name in ARMS:
        epochs = []
        for crop, batch in ((256, 32), (512, 16)):
            r = reports[name][crop]
            train = r['mean_end_to_end_step_seconds_after_warmup'] * (13500 // batch)
            val = r['full512_eval_batch32_seconds'] * math.ceil(1500 / 32)
            if not all(math.isfinite(x) and x > 0 for x in (train, val)):
                raise ValueError('Invalid benchmark timing')
            epochs.append(train + val + 3)
        costs[name] = epochs
    for epochs in (450, 300, 200):
        estimate = 1.3 * max(epochs * a + 40 * b for a, b in costs.values()) + 1800
        if estimate < available_seconds:
            return {'base_epochs': epochs, 'final_epochs': 40,
                    'estimated_seconds_with_margin': estimate,
                    'epoch_seconds_by_arm': costs, 'remaining_seconds': available_seconds,
                    'margin_factor': 1.3, 'evaluation_reserve_seconds': 1800}
    raise RuntimeError('Even 200+40 epochs cannot conservatively fit; no scientific fits started')


def run_child(cmd, gpu, log, deadline):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONUNBUFFERED='1')
    save(log.with_suffix('.command.json'), {'argv': cmd, 'gpu': gpu})
    remaining = deadline - time.time()
    if remaining <= 0:
        raise TimeoutError('Study deadline reached')
    with log.open('x') as out:
        subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, env=env,
                       check=True, timeout=remaining)


def validate_checkpoint(path, name, epochs, parent=None, *, seed=0, data_seed=42):
    import torch
    from models import build_model, count_trainable
    ck = torch.load(path, map_location='cpu', weights_only=False)
    ca = ck['args']
    expected = {'model': name, 'epochs': epochs, 'seed': seed, 'data_seed': data_seed,
                'split_seed': 42, 'val_frac': 0.1, 'train_split': 'train', 'views': 2,
                'jsd_lambda': 0.0, 'clip_grad': 0.0, 'norm': 'imagenet',
                'init_from': None if parent is None else str(parent), 'kd_teacher': None}
    if ck['epoch'] != epochs or any(ca.get(k) != v for k, v in expected.items()):
        raise ValueError(f'Checkpoint identity/epoch mismatch: {path}')
    model = build_model(name)
    model.load_state_dict(ck['model'], strict=True)
    n = count_trainable(model)
    if not 0 < n <= 12_300_000 or ck['n_params_training'] != n:
        raise ValueError('Parameter count mismatch')
    if not all(torch.isfinite(t).all().item() for t in ck['model'].values()):
        raise ValueError('Nonfinite checkpoint')
    if not math.isfinite(ck['val_acc']) or not 0 <= ck['val_acc'] <= 1:
        raise ValueError('Invalid validation score')
    with path.open('rb') as f:
        sha = hashlib.file_digest(f, 'sha256').hexdigest()
    receipt = {'checkpoint': str(path), 'sha256': sha, 'epoch': epochs,
               'model': name, 'parameters': n, 'val_acc': ck['val_acc'], 'status': 'passed'}
    save(path.parent / 'validated.json', receipt)
    return receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--finish-by', default=FINISH_BY)
    args = p.parse_args()
    deadline = datetime.fromisoformat(args.finish_by).timestamp()
    args.output.mkdir(parents=True, exist_ok=False)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1,2,3,4,5,6,7':
        raise ValueError('Expected supervisor allocation of all eight physical GPUs')
    ready = json.loads((args.cache / 'ready.json').read_text())
    from prepare_b200_data import EXPECTED_SHA256
    if ready['archive_sha256'] != EXPECTED_SHA256 or ready['images'] != 15000:
        raise ValueError('Wrong training archive')
    import numpy as np
    from dataset import load_meta, stratified_split
    labels = np.asarray(load_meta(args.cache, 'train')['labels'])
    fit, val = stratified_split(labels, .1, 42)
    if (len(fit), len(val)) != (13500, 1500) or set(fit) & set(val):
        raise ValueError('Invalid split')
    save(args.output / 'split.json', {'fit': fit.tolist(), 'val': val.tolist(), 'split_seed': 42})
    save(args.output / 'manifest.json', {'arms': ARMS, 'seed': 0, 'data_seed': 42,
         'finish_by': args.finish_by, 'source_sha256': {
             x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted(SRC.glob('*.py'))},
         'dataset': ready, 'inference': 'one model, one pass; no TTA or ensemble',
         'selection': 'fixed final endpoint; clean and nine processed validation conditions reported separately',
         'status': 'benchmarking'})
    reports = {name: {} for name in ARMS}

    def bench(item):
        gpu, name = item
        for crop, batch in ((256, 32), (512, 16)):
            dest = args.output / f'benchmark_{name}_{crop}'
            run_child([sys.executable, '-u', str(SRC / 'benchmark_deadline_candidates.py'),
                       '--model', name, '--cache', str(args.cache), '--output', str(dest),
                       '--crop', str(crop), '--batch-size', str(batch)],
                      gpu, args.output / f'benchmark_{name}_{crop}.log', deadline)
            report = json.loads((dest / 'result.json').read_text())
            if report['steps'] != 30 or not report['full512_reload_exact']:
                raise ValueError('Incomplete numerical smoke')
            reports[name][crop] = report
    with ThreadPoolExecutor(max_workers=5) as pool:
        list(pool.map(bench, enumerate(ARMS)))
    budget = select_budget(reports, deadline - time.time())
    save(args.output / 'schedule.json', budget)
    print('SHARED_SCHEDULE', json.dumps(budget), flush=True)

    def fit_arm(item):
        gpu, name = item
        base, final = (args.output / f'{name}_{stage}' for stage in ('base', 'full512'))
        if base.exists() or final.exists():
            raise FileExistsError('Training output already exists')
        run_child(training_command(name, args.cache, base, budget['base_epochs']),
                  gpu, args.output / f'{name}_base.log', deadline)
        base_result = validate_checkpoint(base / 'last.pt', name, budget['base_epochs'])
        run_child(training_command(name, args.cache, final, 40, final=True, parent=base / 'last.pt'),
                  gpu, args.output / f'{name}_full512.log', deadline)
        final_result = validate_checkpoint(final / 'last.pt', name, 40, parent=base / 'last.pt')
        run_child([sys.executable, '-u', str(SRC / 'evaluate_deadline.py'),
                   '--checkpoint', str(final / 'last.pt'), '--cache', str(args.cache),
                   '--output', str(args.output / f'{name}_evaluation')],
                  gpu, args.output / f'{name}_evaluation.log', deadline)
        scores = json.loads((args.output / f'{name}_evaluation/results.json').read_text())
        return {'base': base_result, 'full512': final_result, 'evaluation': scores}
    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(fit_arm, enumerate(ARMS)))
    save(args.output / 'complete.json', {'status': 'passed', 'schedule': budget,
         'results': dict(zip(ARMS, results)), 'finished_utc': datetime.now(timezone.utc).isoformat()})


if __name__ == '__main__':
    main()
