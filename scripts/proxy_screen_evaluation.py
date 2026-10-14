"""Fixed six-checkpoint downstream screen, reusing the existing harness pipeline."""
import argparse
import json
from pathlib import Path
import sys
import time

from eval import suite
from eval.benchmarks import local_dataset_paths, resolve_tasks
from scripts.check_downstream_eval import save

ARMS = ('A', 'P6-iso', 'P6-iso-deep', 'P6-iso-adapter', 'P7-simple', 'P7-simple-lm')
PROTOCOLS = {
    'grammar': (0, ['blimp']),
    'zero': (0, ['sciq', 'lambada']),
    'five': (5, ['piqa', 'arc_easy', 'winogrande', 'boolq']),
    'ten': (10, ['hellaswag']),
    'twentyfive': (25, ['arc_challenge']),
}


def read(path):
    return json.loads(Path(path).read_text())


def predecessor_ready(root, guard=Path('/mnt/local/_gpu_guard/DISABLED')):
    """Require the follow-up's complete success and verified final burn handoff."""
    root = Path(root) / 'supervised'
    if not (root / 'supervisor.json').exists():
        return False
    state = read(root / 'supervisor.json')
    if state.get('training_status') == 'failed':
        raise RuntimeError('Predecessor failed; evaluation has not reclaimed GPUs')
    if not state.get('finished_at'):
        return False
    from scripts.proxy_overnight_queue import CASES
    run = read(root / 'run/run.json')
    burn = read(root / 'burn-verified.json')
    required = {f'{label}-{kind}' for label, _, _ in CASES for kind in ('train', 'validate')}
    passed = {j['name'] for j in run['jobs'] if j['status'] == 'ok'}
    if (state.get('training_status') != 'ok' or state.get('training_returncode') != 0
            or run['status'] != 'ok' or not required.issubset(passed)
            or not (root / 'run/complete.json').is_file()
            or read(root / 'run/summary.json')['status'] != 'passed'
            or not burn.get('collective_progress_verified')
            or not state.get('burn', {}).get('collective_progress_verified')
            or burn.get('session') != state['burn'].get('session')):
        raise RuntimeError('Predecessor completion or burn verification is invalid')
    return not guard.exists()


def make_jobs(scope_path):
    jobs = []
    for stage in ('numerics', 'smoke', 'full'):
        jobs.append(dict(name=stage, gpus=list(range(8)), timeout_seconds=43200,
            argv=['{python}', '-u', '-m', 'scripts.proxy_screen_evaluation', stage,
                  '--scope', str(scope_path), '--root', '{run_dir}/' + stage],
            required_outputs=[dict(path=stage + '/summary.json', json_equals={'status': 'passed'})]))
    return jobs


def audit_zero_shot(mapping, tokenizer, output):
    from transformers import AutoTokenizer
    from lm_eval.tasks import get_task_dict
    from eval.benchmarks import task_configs, benchmark_task_manager
    tok = AutoTokenizer.from_pretrained(tokenizer, local_files_only=True)
    names = resolve_tasks([t for shots, ts in PROTOCOLS.values() if not shots for t in ts], True)
    tasks = get_task_dict([dict(c, num_fewshot=0) for c in task_configs(names, mapping)],
                          task_manager=benchmark_task_manager())
    reports = {}
    for name, task in tasks.items():
        task.set_fewshot_seed(42)
        task.build_all_requests()
        maximum = 0
        for request in task.instances:
            if request.request_type != 'loglikelihood':
                raise ValueError('Expected likelihood-only evaluation')
            context, continuation = request.args
            # Empty-context BLiMP requests receive one harness prefix token.
            total = len(tok.encode(context + continuation, add_special_tokens=False)) + int(not context)
            maximum = max(maximum, total - 1)
        if not task.instances or maximum > 2048:
            raise ValueError('Empty or overlength task: ' + name)
        reports[name] = dict(requests=len(task.instances), max_input_tokens=maximum)
    save(output, dict(status='passed', tasks=reports, max_length=2048, num_fewshot=0))


def prepare(root, host, predecessor, screen):
    import importlib.metadata as md
    import shutil
    import socket
    import subprocess
    from accelerate.commands.config.config_args import default_yaml_config_file, load_config_from_file
    from scripts.train_then_burn import digest, approved_launcher, APPROVED_BURNS, BURN_HASH, GUARD_HASH
    from scripts.verified_gpu_reclaim import inspect, process
    from run_experiments import load_jobs
    from scripts.check_downstream_eval import check_fewshot
    root = Path(root); project = Path.cwd()
    if socket.gethostname() != host:
        raise ValueError('Wrong target host')
    for attempt in range(1440):
        if predecessor_ready(predecessor):
            break
        if attempt % 10 == 0:
            print('WAITING_FOR_TRAINING_AND_VERIFIED_BURNS', attempt, flush=True)
        time.sleep(30)
    else:
        raise RuntimeError('Predecessor wait timed out; GPUs untouched')
    packages = dict(torch='2.14.1', transformers='5.9.0', accelerate='1.13.0', datasets='4.8.5',
                    lm_eval='0.4.10', **{'flash-attn-4': '4.0.0b33', 'nvidia-cutlass-dsl': '4.8.0'})
    versions = {p: md.version(p) for p in packages}
    if any(versions[p].split('+')[0] != v for p, v in packages.items()):
        raise ValueError(('Unexpected environment', versions))
    names = set(resolve_tasks([t for _, ts in PROTOCOLS.values() for t in ts], True))
    source = read(project / 'resources/expanded_inputs_20261012.json')
    selected = []
    for repo in source['repositories']:
        tasks = [t for t in repo['tasks'] if t in names]
        if tasks:
            selected.append({**repo, 'tasks': tasks, 'load_configs': {t: repo['load_configs'][t] for t in tasks}})
    manifest = root / 'manifest.json'
    save(manifest, {**source, 'repositories': selected})
    data = '/mnt/local/_data/deep-llms_th2/expanded-20261012'
    mapping = local_dataset_paths(data, manifest)
    if set(mapping) != names:
        raise ValueError('Benchmark manifest does not cover the exact suite')
    checkpoints = []; specs = {}; fingerprints = None; tokenizer = None
    for arm in ARMS:
        folder = Path(screen) / 'supervised/run/seed-42' / arm
        ck = folder / 'checkpoint-2500'; cfg = read(folder / 'train_config.json')
        result = read(folder / 'result.json')
        if (cfg['pilot']['arm'] != arm or cfg['model_config']['num_hidden_layers'] != 12
                or cfg['training']['seed'] != 42 or cfg['training']['data_seed'] != 42
                or cfg['pilot']['attention_backend'] != 'fa4' or not cfg['training']['bf16']
                or cfg['pilot']['proxy_lambda_max'] != .1
                or read(ck / 'trainer_state.json')['global_step'] != 2500
                or result['global_step'] != 2500 or result['status'] != 'complete'):
            raise ValueError('Unexpected source checkpoint: ' + arm)
        pair = (cfg['train_fingerprint'], cfg['eval_fingerprint'])
        if fingerprints is None:
            fingerprints = pair; tokenizer = cfg['model']['tokenizer_name']
        if pair != fingerprints or tokenizer != cfg['model']['tokenizer_name']:
            raise ValueError('Source data/tokenizer mismatch')
        checkpoints.append(str(ck))
        specs[str(ck)] = dict(arm=arm, step=2500, checkpoint_sha256=digest(ck / 'model.safetensors'))
    # Reuse the established prompt audit. Demonstrations must come from train,
    # and every requested few-shot prompt must fit without left truncation.
    for group, (shots, tasks) in PROTOCOLS.items():
        if not shots:
            continue
        dest = root / ('audit-' + group + '.json')
        check_fewshot(argparse.Namespace(dataset_root=data, dataset_manifest=str(manifest),
            tokenizer=tokenizer, tasks=resolve_tasks(tasks, True), num_fewshot=shots,
            seed=42, max_length=2048, smoke_limit=2, output=str(dest)))
        if any(t['truncated_requests'] for t in read(dest)['tasks'].values()):
            raise ValueError('Few-shot prompts exceed trained context; do not silently change the protocol')
    audit_zero_shot(mapping, tokenizer, root / 'audit-zero.json')
    source = project / 'resources/accelerate_config.yaml'; destination = Path(default_yaml_config_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.copy2(destination, root / 'accelerate.previous.yaml')
    shutil.copy2(source, destination)
    ac = load_config_from_file(str(destination)).to_dict()
    if source.read_bytes() != destination.read_bytes() or ac['num_processes'] != 8 or ac['mixed_precision'] != 'bf16':
        raise ValueError('Accelerate configuration differs')
    subprocess.run(['accelerate', 'env'], check=True)
    status = inspect()
    if status['host'] != host or status['guard_disabled'] or len(status['gpus']) != 8 or any('B200' not in g['name'] for g in status['gpus']):
        raise ValueError('Unexpected GPU ownership/state')
    if digest('/mnt/local/_gpu_guard/gpu_guard.sh') != GUARD_HASH:
        raise ValueError('Unknown GPU guard')
    approved = {**APPROVED_BURNS, str(project / 'resources/llm_pretrain_burn.py'): BURN_HASH}
    if any(not (approved_launcher(p, approved) or approved_launcher(process(p)['ppid'], approved)) for p in status['workers']):
        raise ValueError('Unknown GPU workload; no signals sent')
    scope = root / 'scope.json'
    save(scope, dict(checkpoints=checkpoints, specs=specs, protocols=PROTOCOLS, data_root=data,
        manifest=str(manifest), manifest_sha256=digest(manifest), versions=versions,
        predecessor=str(predecessor), source_screen=str(screen), seed=42,
        paired_loss=False, finetuning=False))
    save(root / 'jobs.json', dict(jobs=make_jobs(scope)))
    load_jobs(root / 'jobs.json')
    save(root / 'inspection.json', status)
    save(root / 'preflight.json', dict(status='passed', versions=versions,
        accelerate_cache=str(destination), accelerate_sha256=digest(destination),
        checkpoints=specs, manifest_sha256=digest(manifest)))
    print('SCREEN_EVALUATION_PREFLIGHT_PASSED', flush=True)


def run(stage, root, scope_path):
    from eval.models import file_hash
    root = Path(root); root.mkdir(parents=True, exist_ok=False)
    scope = read(scope_path); checkpoints = scope['checkpoints']
    if len(checkpoints) != 6 or [scope['specs'][p]['arm'] for p in checkpoints] != list(ARMS):
        raise ValueError('Wrong checkpoint shortlist')
    if scope['protocols'] != {k: [n, ts] for k, (n, ts) in PROTOCOLS.items()}:
        raise ValueError('Changed task protocol')
    if file_hash(scope['manifest']) != scope['manifest_sha256']:
        raise ValueError('Data manifest changed')
    if stage == 'numerics':
        jobs = [dict(output=str(root / f'{i}.json'), argv=[sys.executable, '-m', 'scripts.check_downstream_eval',
            'numerics', '--checkpoint', ck, '--expected-step', '2500', '--output', str(root / f'{i}.json')])
            for i, ck in enumerate(checkpoints)]
        suite.execute(jobs, 8)
        reports = [read(root / f'{i}.json') for i in range(len(checkpoints))]
        for ck, report in zip(checkpoints, reports):
            if report['status'] != 'passed' or any(report[k] != v for k, v in scope['specs'][ck].items()):
                raise ValueError('Checkpoint numerical acceptance failed')
        save(root / 'summary.json', dict(status='passed', checkpoints=reports))
        return
    limit = 2 if stage == 'smoke' else None
    jobs = suite.commands(checkpoints, root, scope['data_root'], scope['manifest'], 'fa4', limit, PROTOCOLS)
    save(root / 'plan.json', jobs)
    suite.execute(jobs, 8)
    result = suite.collect(root, checkpoints, limit, PROTOCOLS)
    for ck, report in result['checkpoints'].items():
        if report['checkpoint_sha256'] != scope['specs'][ck]['checkpoint_sha256']:
            raise ValueError('Checkpoint changed since preflight')
    result['labels'] = {ck: scope['specs'][ck]['arm'] for ck in checkpoints}
    save(root / 'summary.json', result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('prepare', 'numerics', 'smoke', 'full'))
    p.add_argument('--root', required=True); p.add_argument('--scope')
    p.add_argument('--host'); p.add_argument('--predecessor'); p.add_argument('--screen')
    a = p.parse_args()
    if a.stage == 'prepare':
        if not all((a.host, a.predecessor, a.screen)):
            p.error('Preparation needs host, predecessor and screen')
        prepare(a.root, a.host, a.predecessor, a.screen)
    else:
        if not a.scope:
            p.error('Evaluation needs a frozen scope')
        run(a.stage, a.root, a.scope)
