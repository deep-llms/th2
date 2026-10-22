"""Two fresh 10k fits using the immutable, validated router-study source."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

ARMS = ('D', 'Partial-25-cosine')
STEPS = 10000
SOURCE_HASH = '65afe4d86c3e154638d512f2e63c38303d9659b57ef1c4ebf96ff7e857c096c9'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def verify_source(source):
    manifest = source / 'SOURCE_MANIFEST.json'
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != SOURCE_HASH:
        raise ValueError('Unrecognized source snapshot')
    for name, expected in read(manifest)['files'].items():
        path = source / name
        if (path.is_symlink() or not path.resolve().is_relative_to(source.resolve())
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
            raise ValueError('Immutable source changed: ' + name)


def replace_strings(value, replacements):
    if isinstance(value, dict):
        return {k: replace_strings(v, replacements) for k, v in value.items()}
    if isinstance(value, list):
        return [replace_strings(v, replacements) for v in value]
    if isinstance(value, str):
        for before, after in replacements:
            value = value.replace(before, after)
    return value


def prepare(output, source, dense_recipe, router_recipe, driver):
    old = {'D': read(dense_recipe), 'Partial-25-cosine': read(router_recipe)}
    differences = {k for k in old['D'].keys() | old['Partial-25-cosine'].keys()
                   if old['D'].get(k) != old['Partial-25-cosine'].get(k)}
    if differences - {'config_name', 'output_dir', 'keep_checkpoint_steps'}:
        raise ValueError('Unmatched parent training recipes: ' + str(differences))
    models = {arm: read(t['config_name']) for arm, t in old.items()}
    expected = dict(seed=0, data_seed=0, max_steps=4768, warmup_steps=96,
                    block_size=1024, per_device_train_batch_size=16,
                    gradient_accumulation_steps=4, attention_backend='fa4',
                    learning_rate=3e-4, lr_scheduler_type='cosine_with_min_lr',
                    lr_scheduler_kwargs={'min_lr_rate': .1}, save_steps=250,
                    save_total_limit=None, gradient_checkpointing=False)
    for arm, t in old.items():
        if (any(t.get(k) != v for k, v in expected.items())
                or t.get('stop_after') is not None or t.get('resume_from_checkpoint') is not None
                or t.get('model_name_or_path') is not None or models[arm]['bdot']['seed'] != 0):
            raise ValueError('Unexpected parent recipe: ' + arm)
    d, p = models['D']['bdot'], models['Partial-25-cosine']['bdot']
    if (d['routing'] != 'fixed' or d['r'] != 0 or not d.get('step_film', True)
            or p['route_scope'] != 'mlp' or p['routed_mlp_fraction'] != .25
            or p['router_metric'] != 'cosine' or p.get('router_detach_input', False)
            or p['routing'] != 'learned' or p['curriculum'] != 'A'
            or p.get('r', 1) != 1 or p.get('widen_every_frac', .1) != .1):
        raise ValueError('Unexpected D or Partial-25-cosine architecture')
    templates = read(router_recipe.parent / 'jobs.json')['jobs']
    name0 = 'Partial-25-cosine-s0'
    pair = [j for j in templates if j['name'] in (name0, name0 + '-eval')]
    if [j['name'] for j in pair] != [name0, name0 + '-eval']:
        raise ValueError('Missing parent train/eval pair')
    prepared = output / 'prepared'
    prepared.mkdir(exist_ok=False)
    common = ['{python}', '-m', 'accelerate.commands.launch', '--config_file',
              str(source / 'resources/accelerate_b200.yaml'), '--num_processes', '8',
              '--main_process_port', '0']
    jobs = []
    for arm in ARMS:
        t = old[arm]
        jobs.append(dict(name=arm+'-carry', gpus=list(range(8)), timeout_seconds=1800,
            argv=[*common, str(source/'eval/packed_ppl.py'), '--checkpoint',
                  str(Path(t['output_dir'])/'checkpoint-4768'), '--eval_data_dir', t['eval_data_dir'],
                  '--tokenizer_name', t['tokenizer_name'], '--output_file',
                  '{run_dir}/'+arm+'-carry/result.json', '--attention_backend', 'fa4',
                  '--precision', 'bf16', '--batch_size', '16', '--routing_seed', '0'],
            required_outputs=[dict(path=arm+'-carry/result.json', json_equals=dict(status='ok'))]))
    def control(action):
        return dict(name='long_pair_'+action, timeout_seconds=300,
            argv=['{python}', str(driver), action, '--output', str(output)],
            required_outputs=[dict(path='long-pair-'+action+'.json', json_equals=dict(status='ok'))])
    jobs.append(control('check'))
    for arm in ARMS:
        name = arm + '-s0'
        config = prepared / (name + '.model.json')
        write(config, models[arm])
        t = dict(old[arm], max_steps=STEPS, warmup_steps=200,
                 config_name=str(config), output_dir=str(prepared/'runs'/name))
        # Keep every existing 250-step save, including optimizer and all-rank RNG.
        t.pop('keep_checkpoint_steps', None)
        write(prepared / (name + '.train.json'), t)
        for template in pair:
            job = replace_strings(copy.deepcopy(template), [
                (str(router_recipe.parent), str(prepared)), (name0, name),
                ('checkpoint-4768', 'checkpoint-10000')])
            if job['name'] == name:
                index = job['argv'].index('--steps')
                if job['argv'][index + 1] != '4768':
                    raise ValueError('Unexpected parent completion step')
                job['argv'][index + 1] = str(STEPS)
                job['timeout_seconds'] = 14400
                if (job['argv'][-1] != str(prepared/(name+'.train.json'))
                        or str(source/'train.py') not in job['argv']):
                    raise ValueError('Training command does not use frozen source/new recipe')
            elif str(source/'eval/packed_ppl.py') not in job['argv']:
                raise ValueError('Evaluation command does not use frozen source')
            if job['gpus'] != list(range(8)):
                raise ValueError('Expected all eight GPUs per job')
            jobs.append(job)
    jobs.append(control('report'))
    write(output/'long-pair.json', dict(source=str(source), source_manifest_sha256=SOURCE_HASH,
        parent_recipes={a: str(p) for a, p in zip(ARMS, [dense_recipe, router_recipe])},
        steps=STEPS, warmup_steps=200, seed=0, data_seed=0,
        input_tokens=STEPS*524288, checkpoint_policy='all saves every 250 steps'))
    path = output/'gated-jobs.json'
    write(path, dict(jobs=jobs))
    return path


def check(output):
    meta = read(output/'long-pair.json')
    results, data = {}, []
    for arm, recipe in meta['parent_recipes'].items():
        root = Path(read(recipe)['output_dir'])
        result = read(output/'supervised/run'/(arm+'-carry/result.json'))
        if (result['status'] != 'ok' or result['global_step'] != 4768
                or not math.isfinite(result['loss'])
                or abs(result['loss']-read(root/'final_eval.json')['eval_loss']) > 1e-5
                or (result['num_tokens'], result['num_sequences']) != (9915698, 9704)):
            raise ValueError('Checkpoint carry evaluation failed: ' + arm)
        results[arm] = result
        data.append(read(root/'data_recipe.json'))
    if data[0] != data[1]:
        raise ValueError('Parent data orders differ')
    write(output/'supervised/run/long-pair-check.json', dict(status='ok', results=results))


def report(output):
    meta = read(output/'long-pair.json')
    records = []
    reference = read(Path(read(meta['parent_recipes']['D'])['output_dir'])/'data_recipe.json')
    for arm in ARMS:
        name = arm+'-s0'; root = output/'prepared/runs'/name; run = output/'supervised/run'
        result, receipt = read(run/name/'result.json'), read(run/(name+'.verified.json'))
        final, train = read(root/'final_eval.json'), read(root/'train_results.json')
        if (result['status'] != 'ok' or receipt['status'] != 'ok'
                or result['global_step'] != STEPS or receipt['global_step'] != STEPS
                or train['global_step'] != STEPS or train['schedule_steps'] != STEPS
                or train['train_input_tokens'] != STEPS*524288
                or (result['num_tokens'], result['num_sequences']) != (9915698, 9704)
                or not math.isfinite(result['loss']) or abs(result['loss']-final['eval_loss']) > 1e-5
                or not math.isfinite(result['perplexity'])
                or abs(result['perplexity']-math.exp(result['loss'])) > 1e-8
                or read(root/'data_recipe.json') != reference
                or {p.name for p in root.glob('checkpoint-*')} !=
                   {f'checkpoint-{s}' for s in range(250, STEPS+1, 250)}):
            raise ValueError('Invalid final result/checkpoint retention: ' + name)
        records.append(dict(arm=arm, loss=result['loss'], perplexity=result['perplexity'],
                            train_runtime=train['train_runtime']))
    write(run/'long-pair-report.json', dict(status='ok', steps=STEPS, seed=0, runs=records,
        routed_gap_to_D=records[1]['loss']-records[0]['loss'],
        scope='Fresh matched 10k schedules, equal tokens; not an equal-wall-time comparison.'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['launch', 'check', 'report'])
    p.add_argument('--output', type=Path, required=True)
    for name in ['source', 'dense-recipe', 'router-recipe', 'predecessor']:
        p.add_argument('--'+name, type=Path)
    p.add_argument('--host'); p.add_argument('--session')
    a = p.parse_args()
    if a.action == 'check':
        check(a.output); return
    if a.action == 'report':
        report(a.output); return
    if (not all([a.source, a.dense_recipe, a.router_recipe, a.predecessor, a.host, a.session])
            or socket.gethostname() != a.host):
        raise ValueError('Explicit matching host, source, recipes and predecessor required')
    verify_source(a.source)
    sys.path.insert(0, str(a.source))
    from scripts.native_baseline_pipeline import burn_gate
    from scripts.gpu_process import inspect
    from scripts.train_then_burn import burn_progress
    from run_experiments import load_jobs
    initial = read(a.output/'preflight-initial.json')
    if (initial['status'] != 'ok' or initial['live']['host'] != a.host
            or read(a.predecessor/'supervised/run/confirmation-results.json')['status'] != 'ok'
            or read(a.predecessor/'supervised/run/run.json')['status'] != 'ok'
            or not burn_progress((a.predecessor/'supervised/burn.log').read_text())
            or shutil.disk_usage(a.output).free < 700*1024**3):
        raise ValueError('Predecessor, preflight or free disk check failed')
    burn_gate(a.predecessor)
    jobs = prepare(a.output, a.source, a.dense_recipe, a.router_recipe, Path(__file__).resolve())
    load_jobs(jobs)
    burn_gate(a.predecessor)
    write(a.output/'preflight.json', inspect())
    subprocess.run([sys.executable, str(a.source/'scripts/bdot_b200_launch.py'),
        '--root', str(a.output), '--host', a.host, '--session', a.session,
        '--jobs-config', str(jobs), '--predecessor-burn-script', str(a.source/'resources/llm_pretrain_burn.py')],
        cwd=a.source, check=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7'})


if __name__ == '__main__':
    main()
