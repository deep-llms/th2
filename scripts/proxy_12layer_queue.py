"""Prepare (never launch) the nine-arm, PPL-only 12-layer screen.

Uses train.py / Accelerate / run_experiments unchanged. The historical 28-layer
recipe is separate; this is a full 2,500-step schedule with 5% warmup.
"""
import argparse
import json
from pathlib import Path

from deep_kv.__main__ import jobs
from run_experiments import load_jobs


ARMS = ('A', 'P6-iso', 'P6-iso-targetgrad', 'P6-iso-shallowgrad',
        'P6-iso-bothgrad', 'P6', 'P6-lowaux', 'P6-iso-sparse', 'P7-simple')


def make(recipe_path, output, seed=42):
    recipe = json.loads(Path(recipe_path).read_text())
    expected = dict(num_hidden_layers=12, consumer=2, deep_target=12,
                    max_steps=2500, stop_after=2500, warmup_steps=125,
                    block_size=2048, isolate_documents=True, proxy_screen=True,
                    per_device_train_batch_size=16, gradient_accumulation_steps=4,
                    save_steps=250, save_total_limit=0, attention_backend='fa4',
                    checkpoint_layers=False, checkpoint_lm=False, checkpoint_aux=False,
                    proxy_sync_optimization=True)
    if any(recipe.get(key) != value for key, value in expected.items()):
        raise ValueError('12-layer screen recipe differs from its fixed protocol')
    if (type(seed) is not int or seed < 0 or recipe.get('proxy_lambda_max') is not None
            or recipe.get('proxy_module_seed') is not None or recipe.get('resume_from_checkpoint')):
        raise ValueError('Use fresh runs and arm-defined lambda / per-seed proxy initialization')
    generated = jobs(recipe_path, arms=ARMS, seeds=[seed])['jobs']
    items = []
    for item in generated:
        if 'gpus' not in item:
            items.append(item)
            continue
        argv = item['argv']
        arm = argv[argv.index('--arm') + 1]
        if arm != 'A':
            argv += ['--proxy_module_seed', str(seed + 1)]
        item['argv'] = ['env', f'PYTHONHASHSEED={seed}', *argv]
        items.append(item)
        name = f'seed-{seed}-validate-{arm}'
        items.append(dict(name=name, argv=[
            '{python}', '-m', 'scripts.check_fa4_proxy', 'validate',
            '--run-dir', '{run_dir}', '--steps', '2500', '--seed', str(seed),
            '--schedule-steps', '2500', '--warmup-steps', '125',
            '--num-hidden-layers', '12', '--arms', arm,
            '--attention-backend', 'fa4', '--output', '{run_dir}/'+name+'.json'],
            required_outputs=[dict(path=name+'.json', json_equals={'status':'passed'})]))
    with Path(output).open('x') as handle:
        json.dump(dict(jobs=items), handle, indent=2)
    load_jobs(output)
    return items


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', default='proxy_heads.12layer.b200.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    make(args.recipe, args.output, args.seed)
