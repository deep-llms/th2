"""Fresh 10k D/control exit adaptation, assessed at 5k, with native HF resume."""
import argparse
import hashlib
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_stop_adaptation import initialize, read, write
from scripts.bdot_stop_launch import snapshot
from scripts.native_baseline_pipeline import burn_gate
from scripts.gpu_process import inspect

OVERLAYS = ('scripts/prepare_stop_adaptation.py', 'eval/stop_v2_study.py',
            'scripts/stop_long_launch.py', 'scripts/stop_long_eval.py',
            'scripts/run_checked_training.py', 'scripts/training_results.py',
            'tests/bdot/test_stop_long.py', 'tests/bdot/test_stop_adaptation.py')
STEPS = 10000
MIDPOINT = 5000
START_SEQUENCE = 5120000
FIT_START = START_SEQUENCE + STEPS * 512


def prepare(parent, previous, root, source):
    prepared = root / 'prepared'
    prepared.mkdir()
    old = read(parent / 'prepared/D-s0.train.json')
    expected = dict(max_steps=10000, block_size=1024, seed=0, data_seed=0,
                    per_device_train_batch_size=16, gradient_accumulation_steps=4,
                    train_sampling_strategy='sequential', dataloader_drop_last=True,
                    attention_backend='fa4', gradient_checkpointing=False,
                    save_steps=250, save_total_limit=None)
    if any(old.get(k) != v for k, v in expected.items()):
        raise ValueError('Unexpected parent recipe')
    if old.get('resume_from_checkpoint') or old.get('stop_after') or old.get('warmup_ratio') is not None:
        raise ValueError('Parent overrides adaptation schedule/resume')
    cp = Path(old['output_dir']) / 'checkpoint-10000'
    if old.get('lr_scheduler_kwargs') != {'min_lr_rate': .1}:
        raise ValueError('Expected the established cosine floor')
    branches = {}
    for name, auxiliary in [('D-continue', None), ('D-DualCE6', dict(depth=6, ce_weight=.5, kl_weight=0.))]:
        init = prepared / (name + '-init')
        initialize(cp, init, 0., stop_settings=dict(min_steps=6, exit_steps=[6]),
                   exit_training=auxiliary, schedule_steps=STEPS)
        run = prepared / 'runs' / name
        branches[name] = str(run / f'checkpoint-{STEPS}')
        base = dict(old, model_name_or_path=str(init), config_name=str(init), output_dir=str(run),
                    max_steps=STEPS, warmup_steps=200, learning_rate=3e-5,
                    train_start_sequence=START_SEQUENCE, train_parent_checkpoint=str(cp))
        base.pop('keep_checkpoint_steps', None)
        for end in (MIDPOINT, STEPS):
            config = dict(base, stop_after=end)
            if end == STEPS:
                config['resume_from_checkpoint'] = str(run / f'checkpoint-{MIDPOINT}')
            write(prepared / f'{name}-{end}.train.json', config)

    write(root / 'branches.json', branches)
    manifest = read(previous / 'evaluation_manifest.json')
    smoke = {name: manifest[name]['checkpoint'] for name in branches}
    write(root / 'smoke_checkpoints.json', smoke)
    for end in (MIDPOINT, STEPS):
        write(root / f'eval-{end}.json', {name: str(Path(cp).parent / f'checkpoint-{end}')
                                         for name, cp in branches.items()})
    write(root / 'adaptation.json', dict(parent=str(parent), predecessor=str(previous),
          steps=STEPS, midpoint=MIDPOINT, start_sequence=START_SEQUENCE, next_sequence=FIT_START,
          head_training_rows=[FIT_START, FIT_START + 256],
          calibration_rows=[FIT_START + 256, FIT_START + 512],
          input_tokens_per_branch=STEPS * 524288, peak_lr=3e-5, warmup=200, min_lr=3e-6,
          branches=branches, optimizer='Fresh AdamW from original 10k parent; native resume at midpoint',
          objective='Control full LM; DualCE6 full LM + 0.5 exit CE, unchanged architecture/loss',
          assessment='Report 5k then continue authorized 10k; numerical/artifact failures stop the queue'))
    jobs = []

    def job(name, command, output, timeout=2400, status='ok'):
        jobs.append(dict(name=name, argv=command, gpus=list(range(8)), timeout_seconds=timeout,
                         required_outputs=[dict(path=output, json_equals=dict(status=status))]))

    job('capacity', ['{python}', str(source / 'scripts/check_exit_training_gpu.py'),
                    '--root', str(root), '--output', '{run_dir}/capacity.json'], 'capacity.json')
    resume_config = read(source / 'configs/bdot/D.json')
    resume_config['bdot']['curriculum'] = 'none'
    resume_config['bdot_stop'] = dict(train_random_prob=0., min_steps=6, exit_steps=[6])
    resume_config['bdot_exit_training'] = dict(depth=6, ce_weight=.5, kl_weight=0.)
    write(root / 'resume-model.json', resume_config)
    job('resume_gate', ['{python}', str(source / 'tests/bdot/resume_gpu_probe.py'), 'suite',
                       '--output', '{run_dir}/resume', '--accelerate-config',
                       str(source / 'resources/accelerate_b200.yaml'), '--steps', '8', '--split', '4',
                       '--soft-split', '2', '--curriculum', 'none',
                       '--variant-config', str(root / 'resume-model.json')], 'resume/summary.json', status='passed')

    def evaluation(name, manifest, step):
        job(name, ['{python}', str(source / 'scripts/stop_long_eval.py'), '--checkpoints', str(manifest),
                   '--parent-recipe', str(parent / 'prepared/D-s0.train.json'), '--step', str(step),
                   '--fit-start', str(FIT_START), '--output', '{run_dir}/' + name], name + '/summary.json', 3600)

    evaluation('eval_smoke', root / 'smoke_checkpoints.json', 1000)
    launch = ['{python}', '-m', 'accelerate.commands.launch', '--config_file',
              str(source / 'resources/accelerate_b200.yaml'), '--num_processes', '8', '--main_process_port', '0']
    for end in (MIDPOINT, STEPS):
        for name in branches:
            recipe = prepared / f'{name}-{end}.train.json'
            receipt = f'checks/{name}-{end}.json'
            job(f'{name}-{end}', ['{python}', str(source / 'scripts/run_checked_training.py'),
                '--arguments', str(recipe), '--receipt', '{run_dir}/' + receipt,
                '--expected-step', str(end), '--start-step', str(0 if end == MIDPOINT else MIDPOINT),
                '--schedule-steps', str(STEPS), '--', *launch, str(source / 'train.py'), str(recipe)],
                receipt, 18000)
        evaluation(f'eval_{end}', root / f'eval-{end}.json', end)
    write(root / 'jobs.json', dict(jobs=jobs))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'parent', 'previous'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--host', required=True)
    parser.add_argument('--session', required=True)
    args = parser.parse_args()
    if socket.gethostname() != args.host:
        raise ValueError('Wrong node')
    burn_gate(args.previous)
    if read(args.previous / 'supervised/run/run.json')['status'] != 'ok':
        raise ValueError('Previous queue did not finish successfully')
    original = args.previous / 'source'
    for rel, sha in read(original / 'SOURCE_MANIFEST.json')['files'].items():
        if hashlib.sha256((original / rel).read_bytes()).hexdigest() != sha:
            raise ValueError('Previous source changed: ' + rel)
    args.root.mkdir(parents=True, exist_ok=False)
    source = args.root / 'source'
    snapshot(original, source, OVERLAYS)
    prepare(args.parent, args.previous, args.root, source)
    burn_gate(args.previous)
    write(args.root / 'preflight.json', inspect())
    subprocess.run([sys.executable, str(source / 'scripts/bdot_b200_launch.py'),
                    '--root', str(args.root), '--host', args.host, '--session', args.session,
                    '--jobs-config', str(args.root / 'jobs.json'),
                    '--predecessor-burn-script', str(original / 'resources/llm_pretrain_burn.py')],
                   cwd=source, check=True)


if __name__ == '__main__':
    main()
