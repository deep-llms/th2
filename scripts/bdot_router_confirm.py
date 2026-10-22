"""Confirm two selected routers at seed 1 using the unchanged, verified source.

Only orchestration lives here. Training, evaluation, checkpoint validation and
GPU handoffs use the immutable source from the completed six-router study.
"""
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

ARMS = ('Partial-25-cosine', 'Partial-50-cosine-detach')
SOURCE_HASH = '65afe4d86c3e154638d512f2e63c38303d9659b57ef1c4ebf96ff7e857c096c9'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


def unchanged(a, b, exceptions=()):
    changes = {k: [a.get(k), b.get(k)] for k in a.keys() | b.keys()
               if k not in exceptions and a.get(k) != b.get(k)}
    if changes:
        raise ValueError('Unrelated recipe changes: ' + str(changes))


def remap(value, replacements):
    if isinstance(value, dict):
        return {k: remap(v, replacements) for k, v in value.items()}
    if isinstance(value, list):
        return [remap(v, replacements) for v in value]
    if isinstance(value, str):
        for before, after in replacements:
            value = value.replace(before, after)
    return value


def verify_source(source):
    manifest = source / 'SOURCE_MANIFEST.json'
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != SOURCE_HASH:
        raise ValueError('Unrecognized source snapshot')
    for name, expected in read(manifest)['files'].items():
        path = source / name
        if (path.is_symlink() or not path.resolve().is_relative_to(source.resolve())
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
            raise ValueError('Immutable source changed: ' + name)


def prepare(output, predecessor, baseline_recipe, driver):
    source = predecessor / 'source'
    base = read(baseline_recipe)
    if (base['seed'], base['data_seed'], base['max_steps'], base['warmup_steps']) != (1, 0, 4768, 96):
        raise ValueError('Unexpected seed-1 dense recipe')
    prepared = output / 'prepared'
    prepared.mkdir(exist_ok=False)
    original_jobs = read(predecessor / 'prepared/jobs.json')['jobs']
    common = ['{python}', '-m', 'accelerate.commands.launch', '--config_file',
              str(source / 'resources/accelerate_b200.yaml'), '--num_processes', '8',
              '--main_process_port', '0']
    carry = dict(name='D-s1-carry', gpus=list(range(8)), timeout_seconds=1800,
        argv=[*common, str(source / 'eval/packed_ppl.py'),
              '--checkpoint', str(Path(base['output_dir']) / 'checkpoint-4768'),
              '--eval_data_dir', base['eval_data_dir'], '--tokenizer_name', base['tokenizer_name'],
              '--output_file', '{run_dir}/D-s1-carry/result.json', '--attention_backend', 'fa4',
              '--precision', 'bf16', '--batch_size', '16'],
        required_outputs=[dict(path='D-s1-carry/result.json', json_equals=dict(status='ok'))])
    def control(action, artifact):
        return dict(name='confirmation_' + action, timeout_seconds=300,
            argv=['{python}', str(driver), action, '--output', str(output)],
            required_outputs=[dict(path=artifact, json_equals=dict(status='ok'))])
    jobs = [carry, control('check', 'confirmation-check.json')]
    for arm in ARMS:
        old_name, name = arm + '-s0', arm + '-s1'
        old = read(predecessor / 'prepared' / (old_name + '.train.json'))
        unchanged(base, old, {'config_name', 'output_dir', 'seed'})
        if old['seed'] != 0:
            raise ValueError('Expected seed-0 parent')
        model = read(old['config_name'])
        if model['bdot']['seed'] != 0:
            raise ValueError('Unexpected parent routing seed')
        model['bdot']['seed'] = 1
        config = prepared / (name + '.model.json')
        write(config, model)
        new = dict(old, seed=1, config_name=str(config), output_dir=str(prepared / 'runs' / name))
        write(prepared / (name + '.train.json'), new)
        selected = [copy.deepcopy(j) for j in original_jobs if j['name'] in (old_name, old_name + '-eval')]
        if [j['name'] for j in selected] != [old_name, old_name + '-eval']:
            raise ValueError('Missing or duplicate original train/eval pair')
        for job in selected:
            job = remap(job, [(str(predecessor / 'prepared'), str(prepared)), (old_name, name)])
            if job['name'].endswith('-eval'):
                idx = job['argv'].index('--routing_seed')
                if job['argv'][idx + 1] != '0':
                    raise ValueError('Unexpected evaluation routing seed')
                job['argv'][idx + 1] = '1'
            jobs.append(job)
    jobs.append(control('report', 'confirmation-results.json'))
    write(output / 'confirmation.json', dict(predecessor=str(predecessor), source=str(source),
        baseline_recipe=str(baseline_recipe), arms=list(ARMS), seed=1, data_seed=0,
        source_manifest_sha256=SOURCE_HASH))
    path = output / 'gated-jobs.json'
    write(path, dict(jobs=jobs))
    return path


def check(output):
    meta = read(output / 'confirmation.json')
    base = read(meta['baseline_recipe'])
    original = read(Path(base['output_dir']) / 'final_eval.json')
    result = read(output / 'supervised/run/D-s1-carry/result.json')
    if (result['status'] != 'ok' or result['global_step'] != 4768
            or not math.isfinite(result['loss']) or abs(result['loss'] - original['eval_loss']) > 1e-5
            or (result['num_tokens'], result['num_sequences']) != (9915698, 9704)):
        raise ValueError('Dense seed-1 checkpoint does not reproduce')
    write(output / 'supervised/run/confirmation-check.json', dict(status='ok', baseline=result))


def report(output):
    meta = read(output / 'confirmation.json')
    prior = Path(meta['predecessor'])
    base1 = Path(read(meta['baseline_recipe'])['output_dir'])
    base0 = Path(read(prior / 'followup.json')['pilot']) / 'prepared/runs/D'
    records = []
    for arm in ARMS:
        losses, gaps = [], []
        for seed, root, base in [(0, prior / 'prepared/runs' / (arm + '-s0'), base0),
                                 (1, output / 'prepared/runs' / (arm + '-s1'), base1)]:
            name = arm + '-s' + str(seed)
            run = prior if seed == 0 else output
            result = read(run / 'supervised/run' / name / 'result.json')
            receipt = read(run / 'supervised/run' / (name + '.verified.json'))
            final = read(root / 'final_eval.json')
            if (result['status'] != 'ok' or receipt['status'] != 'ok'
                    or result['global_step'] != 4768 or receipt['global_step'] != 4768
                    or (result['num_tokens'], result['num_sequences']) != (9915698, 9704)
                    or not math.isfinite(result['loss']) or abs(result['loss'] - final['eval_loss']) > 1e-5
                    or read(root / 'data_recipe.json') != read(base / 'data_recipe.json')):
                raise ValueError('Invalid comparison for ' + name)
            losses.append(result['loss'])
            gaps.append(result['loss'] - read(base / 'final_eval.json')['eval_loss'])
        records.append(dict(arm=arm, seed_losses=losses, seed_gaps_to_D=gaps,
            mean_loss=sum(losses)/2, mean_gap_to_D=sum(gaps)/2,
            within_001_nats_both_seeds=all(g <= .01 for g in gaps)))
    write(output / 'supervised/run/confirmation-results.json', dict(status='ok', arms=records,
        scope='Two initialization/router seeds with fixed data order; not a claim of statistical equivalence or an automatic STOP launch.'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['launch', 'check', 'report'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--predecessor', type=Path)
    p.add_argument('--baseline-recipe', type=Path)
    p.add_argument('--host')
    p.add_argument('--session')
    a = p.parse_args()
    if a.action == 'check':
        check(a.output)
        return
    if a.action == 'report':
        report(a.output)
        return
    if not all([a.predecessor, a.baseline_recipe, a.host, a.session]) or socket.gethostname() != a.host:
        raise ValueError('Explicit matching host, predecessor, baseline and session required')
    source = a.predecessor / 'source'
    verify_source(source)
    sys.path.insert(0, str(source))
    from scripts.native_baseline_pipeline import burn_gate
    from scripts.gpu_process import inspect
    from scripts.train_then_burn import burn_progress
    from run_experiments import load_jobs
    if read(a.output / 'preflight-initial.json')['live']['host'] != a.host:
        raise ValueError('Preflight node differs')
    burn_gate(a.predecessor)
    if (read(a.predecessor / 'supervised/supervisor.json')['training_status'] != 'ok'
            or read(a.predecessor / 'supervised/run/checks/summary.json')['status'] != 'passed'
            or not burn_progress((a.predecessor / 'supervised/burn.log').read_text())
            or shutil.disk_usage(a.output).free < 300*1024**3):
        raise ValueError('Missing verified predecessor/acceptance or free disk space')
    jobs = prepare(a.output, a.predecessor, a.baseline_recipe, Path(__file__).resolve())
    load_jobs(jobs)
    burn_gate(a.predecessor)
    write(a.output / 'preflight.json', inspect())
    subprocess.run([sys.executable, str(source / 'scripts/bdot_b200_launch.py'),
        '--root', str(a.output), '--host', a.host, '--session', a.session,
        '--jobs-config', str(jobs), '--predecessor-burn-script', str(source / 'resources/llm_pretrain_burn.py')],
        cwd=source, check=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7'})


if __name__ == '__main__':
    main()
