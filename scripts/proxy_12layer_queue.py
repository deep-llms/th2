"""Prepare (never launch) the nine-arm, PPL-only 12-layer screen.

Uses train.py / Accelerate / run_experiments unchanged. The historical 28-layer
recipe is separate; this is a full 2,500-step schedule with 5% warmup.
--extensions adds six paired adapter/depth/deep-residual arms, for fifteen total.
--all-arms selects the complete twenty-four-arm screen.
"""
import argparse
import copy
import json
from pathlib import Path

from deep_kv.__main__ import jobs
from deep_kv import P6_EXTENSION_ARMS, FOLLOWUP_ARMS, P6_CONSUMER_ARMS
from run_experiments import load_jobs


ARMS = ('A', 'P6-iso', 'P6-iso-targetgrad', 'P6-iso-shallowgrad',
        'P6-iso-bothgrad', 'P6', 'P6-lowaux', 'P6-iso-sparse', 'P7-simple')
ALL_SCREEN_ARMS = ARMS + P6_EXTENSION_ARMS + FOLLOWUP_ARMS


def make(recipe_path, output, seed=42, extensions=False, all_arms=False, defer_consumer=False):
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
    selected = ALL_SCREEN_ARMS if all_arms else ARMS + (P6_EXTENSION_ARMS if extensions else ())
    if defer_consumer:
        if not all_arms:raise ValueError('Deferring consumer arms requires --all-arms')
        selected = tuple(a for a in selected if a not in P6_CONSUMER_ARMS)
    generated = jobs(recipe_path, arms=selected, seeds=[seed])['jobs']
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


def launch_jobs(recipe_path, output, rows):
    """Fresh 22-arm queue, gated by real-kernel numerics and eight-GPU smokes.

    Smoke outputs use a separate subtree and never become production checkpoints.
    All tokenization, packing, optimization and training still run through train.py.
    """
    output=Path(output)
    production=make(recipe_path,output.with_name('production-jobs.json'),all_arms=True,defer_consumer=True)
    selected=[j['argv'][j['argv'].index('--arm')+1] for j in production if 'gpus' in j]
    # Independent numerical cases share the eight GPUs concurrently. A is
    # covered by its Trainer smoke; all 21 proxy branches get active checks.
    items=[dict(name='numerics',gpus=list(range(8)),argv=[
        '{python}','-u','-m','scripts.check_fa4_proxy','run','--recipe',str(recipe_path),
        '--rows',str(rows),'--worker-arms',*[a for a in selected if a!='A'],
        '--output','{run_dir}/numerics'],
        required_outputs=[dict(path='numerics/summary.json',json_equals={'status':'passed'})])]
    # Preserve production schedule and batch size; exercise lambda=0 and >0.
    # Active delayed branches are covered above; these smokes verify dormancy.
    for original in production:
        if 'gpus' not in original:continue
        item=copy.deepcopy(original);item['name']='smoke-'+item['name']
        item['argv']=[a.replace('{run_dir}','{run_dir}/smoke') for a in item['argv']]
        for key,value in dict(stop_after=3,save_steps=3,logging_steps=1,eval_steps=3,
                              eval_rows=16,monitor_rows=16,eval_on_start=False).items():
            item['argv'][item['argv'].index('--'+key)+1]=str(value).lower() if isinstance(value,bool) else str(value)
        for required in item['required_outputs']:
            required['path']='smoke/'+required['path']
            required['json_equals'].update(global_step=3,status='stopped')
        items.append(item)
        arm=item['argv'][item['argv'].index('--arm')+1]
        validator=copy.deepcopy(next(j for j in production if j['name']=='seed-42-validate-'+arm))
        validator['name']='smoke-'+validator['name']
        validator['argv']=[a.replace('{run_dir}','{run_dir}/smoke') for a in validator['argv']]
        validator['argv'][validator['argv'].index('--steps')+1]='3'
        for required in validator['required_outputs']:required['path']='smoke/'+required['path']
        items.append(validator)
    items.extend(production)
    with output.open('x') as handle:json.dump(dict(jobs=items),handle,indent=2)
    load_jobs(output)
    return items


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', default='proxy_heads.12layer.b200.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--extensions', action='store_true', help='Include the six adapter/depth/deep-residual arms and controls')
    parser.add_argument('--all-arms', action='store_true', help='Include all 24 arms, including delayed introduction and consumer-aware losses')
    parser.add_argument('--defer-consumer', action='store_true', help='Omit the two consumer-aware arms from --all-arms')
    args = parser.parse_args()
    make(args.recipe, args.output, args.seed, args.extensions, args.all_arms, args.defer_consumer)
