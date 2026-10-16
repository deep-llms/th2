"""Four scratch, native512 fits on the explicitly reserved physical GPUs4–7."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

from deadline_study import run_child, save, validate_checkpoint

SRC = Path(__file__).resolve().parent
ARMS = [('band_strong', 'band_stem', 'strong'),
        ('band_moderate', 'band_stem', 'moderate'),
        ('phase_strong', 'phase_stem', 'strong'),
        ('band_ibn_strong', 'band_stem_ibn', 'strong')]


def command(model, recipe, cache, output, epochs):
    cmd = [sys.executable, '-u', str(SRC/'train.py'), '--model', model,
           '--cache-dir', str(cache), '--out-root', str(output.parent),
           '--run-name', output.name, '--epochs', str(epochs),
           '--batch-size', '16', '--val-batch-size', '32', '--workers', '8',
           '--crop', '512', '--rrc-size', '512', '--crop-upscale',
           '--seed', '0', '--data-seed', '42', '--split-seed', '42', '--val-frac', '.1',
           '--views', '2', '--norm', 'imagenet', '--optim', 'adam', '--lr', '.0003',
           '--weight-decay', '.0001', '--label-smoothing', '0', '--clip-grad', '0',
           '--sched', 'cosine', '--warmup-epochs', '0', '--deterministic',
           '--flips', 'rot90_hflip', '--jsd-lambda', '0']
    if recipe == 'strong':
        cmd += ['--social-mode', 'cf', '--social-p', '1', '--social-region',
                '--social-cf-interp', '--grip-blur-p', '.1', '--grip-blur-smax', '3',
                '--rrc-p', '.2', '--sharpen-p', '.5', '--sharpen-mode', 'composite']
    elif recipe == 'moderate':
        # Existing JPEG implementation randomly chooses PIL or OpenCV. Generic
        # quality60–100 for every processed view, plus mild random crop-resize.
        # This is deliberately NOT clean/flips-only training.
        cmd += ['--grip-jpeg-p', '1', '--grip-jpeg-qmin', '60', '--grip-jpeg-qmax', '100',
                '--rrc-p', '.2', '--rrc-scale-min', '.64']
    else:
        raise ValueError(recipe)
    return cmd


def choose_epochs(reports, remaining):
    costs = {name: r['mean_end_to_end_step_seconds_after_warmup'] * (13500//16)
             + r['full512_eval_batch32_seconds'] * math.ceil(1500/32) + 3
             for name, r in reports.items()}
    if any(not math.isfinite(x) or x <= 0 for x in costs.values()):
        raise ValueError('Bad timing')
    for n in (450, 384, 300, 250, 200, 150):
        estimate = max(costs.values()) * n * 1.3 + 1800
        if estimate < remaining:
            return {'epochs': n, 'epoch_seconds': costs, 'estimated_seconds_with_margin': estimate,
                    'margin_factor': 1.3, 'evaluation_reserve_seconds': 1800}
    raise RuntimeError('Fewer than150 epochs fit; do not start an under-budget fit')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--finish-by', default='2026-10-17T02:00:00+00:00')
    args = p.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    deadline = datetime.fromisoformat(args.finish_by).timestamp()
    save(args.output/'manifest.json', {'arms': ARMS, 'scratch': True, 'seed': 0,
         'split_seed': 42, 'data_seed': 42, 'resolution': 512, 'finish_by': args.finish_by,
         'note': 'Same supplied training partition; no dev/test, pretrained weights, TTA or ensemble.'})
    reports = {}
    def bench(item):
        i, (name, model, recipe) = item
        dest = args.output / ('smoke_'+name)
        # Common stronger augmentation is a conservative timing/check gate.
        # The actual moderate recipe also receives a real one-epoch fit below.
        run_child([sys.executable, '-u', str(SRC/'benchmark_deadline_candidates.py'),
                   '--model', model, '--cache', str(args.cache), '--output', str(dest),
                   '--crop', '512', '--batch-size', '16'], i+4,
                  args.output/(name+'_smoke.log'), deadline)
        r = json.loads((dest/'result.json').read_text())
        if r['steps'] != 30 or not r['full512_reload_exact'] or r['parameters'] > 12300000:
            raise ValueError('Incomplete smoke')
        # Real entrypoint, actual recipe: finite epoch/checkpoint/reload before
        # expensive fitting. Smoke states are never used by scientific fits.
        tiny = args.output/('entrypoint_'+name)
        run_child(command(model, recipe, args.cache, tiny, 1), i+4,
                  args.output/(name+'_entrypoint.log'), deadline)
        validate_checkpoint(tiny/'last.pt', model, 1)
        reports[name] = r
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(bench, enumerate(ARMS)))
    schedule = choose_epochs(reports, deadline-time.time())
    save(args.output/'schedule.json', schedule)
    print('SHARED_SCHEDULE', json.dumps(schedule), flush=True)
    def fit(item):
        i, (name, model, recipe) = item
        dest = args.output/name
        run_child(command(model, recipe, args.cache, dest, schedule['epochs']),
                  i+4, args.output/(name+'.log'), deadline)
        checkpoint = validate_checkpoint(dest/'last.pt', model, schedule['epochs'])
        run_child([sys.executable, '-u', str(SRC/'evaluate_deadline.py'),
                   '--checkpoint', str(dest/'last.pt'), '--cache', str(args.cache),
                   '--output', str(args.output/(name+'_evaluation'))], i+4,
                  args.output/(name+'_evaluation.log'), deadline)
        result = json.loads((args.output/(name+'_evaluation')/'results.json').read_text())
        if result['status'] != 'passed':
            raise ValueError('Incomplete evaluation')
        return {'arm': name, 'checkpoint': checkpoint, 'evaluation': result}
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fit, enumerate(ARMS)))
    save(args.output/'complete.json', {'status': 'passed', 'schedule': schedule,
         'results': results, 'finished_utc': datetime.now(timezone.utc).isoformat()})


if __name__ == '__main__':
    main()
