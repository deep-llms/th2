"""Gate a fresh optimized 10k D/DualCE pair after a verified burn handoff."""
import argparse
import copy
import hashlib
import math
from pathlib import Path
import socket
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_stop_adaptation import read, write

EXTRA = ('bdot/exit_training.py', 'scripts/stop_optimized_launch.py',
         'scripts/check_exit_optimization.py', 'scripts/benchmark_bdot.py',
         'tests/bdot/test_exit_optimization.py', 'tests/bdot/ddp_probe.py',
         'scripts/train_then_burn.py', 'tests/test_train_then_burn.py')


def gate(run):
    required = {'numerics.json': 'passed', 'capacity.json': 'ok',
                'resume/summary.json': 'passed', 'ddp/rank0.json': 'passed',
                'ddp/rank1.json': 'passed'}
    for path, status in required.items():
        if read(run / path)['status'] != status:
            raise ValueError('Optimization prerequisite failed: ' + path)
    timings = {}
    device_bytes = min(r['device_total_bytes'] for r in read(run / 'capacity.json')['branches'].values())
    for case in ('control', 'reference', 'optimized'):
        reports = [read(run / f'benchmark-{case}-{repeat}.json') for repeat in (1, 2)]
        for r in reports:
            if (r['status'] != 'completed' or r['world_size'] != 8
                    or r['settings']['batch_size'] != 16 or r['settings']['accumulation'] != 4
                    or r['settings']['length'] != 1024 or r['settings']['attention_backend'] != 'fa4'
                    or r['settings']['expert_backend'] != 'grouped'
                    or r['settings']['checkpointing'] or not math.isfinite(r['seconds_per_update'])
                    or r['seconds_per_update'] <= 0 or not 0 < r['peak_allocated_bytes'] < .85 * device_bytes):
                raise ValueError('Wrong throughput workload')
        timings[case] = dict(seconds_per_update=statistics.mean(r['seconds_per_update'] for r in reports),
                             repeats=[r['seconds_per_update'] for r in reports],
                             peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in reports))
    gain = 1 - timings['optimized']['seconds_per_update'] / timings['reference']['seconds_per_update']
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError('Optimization did not improve measured whole-update throughput')
    write(run / 'optimization-gate.json', dict(status='passed', timings=timings,
          update_time_reduction=gain,
          optimized_over_control=timings['optimized']['seconds_per_update']/timings['control']['seconds_per_update'],
          note='Fresh branch only; mathematical equivalence is not bitwise migration of old training.'))


def prepare_optimized(parent, smoke_previous, root, source):
    from scripts.stop_long_launch import prepare

    jobs = prepare(parent, smoke_previous, root, source, optimized=True)
    jobs[1]['argv'] += ['--expert-backend', 'grouped']
    init = root / 'prepared/D-DualCE6-init'
    optimized = read(init / 'config.json')
    reference = copy.deepcopy(optimized)
    reference['bdot_exit_training'].update(ce_backend='chunked', shared_prefix=False)
    configs = {'control': read(root / 'prepared/D-continue-init/config.json'),
               'reference': reference, 'optimized': optimized}
    for case, config in configs.items():
        write(root / f'benchmark-{case}.json', config)

    def job(name, argv, path, status='passed', timeout=2400):
        return dict(name=name, argv=argv, gpus=list(range(8)), timeout_seconds=timeout,
                    required_outputs=[dict(path=path, json_equals=dict(status=status))])

    numerics = job('optimization_numerics', ['{python}', str(source / 'scripts/check_exit_optimization.py'),
        '--config', str(init / 'config.json'), '--model-name-or-path', str(init),
        '--output', '{run_dir}/numerics.json', '--attention-backend', 'fa4',
        '--batch-size', '2', '--length', '1024'], 'numerics.json')
    ddp = job('optimization_ddp', ['{python}', '-m', 'torch.distributed.run', '--standalone',
        '--nproc_per_node=2', str(source / 'tests/bdot/ddp_probe.py'), '{run_dir}/ddp',
        '--device', 'cuda', '--bf16', '--exit-training', '--exit-shared-prefix', '--exit-native-ce',
        '--expert-backend', 'grouped'], 'ddp/rank0.json')
    launch = ['{python}', '-m', 'accelerate.commands.launch', '--config_file',
              str(source / 'resources/accelerate_b200.yaml'), '--num_processes', '8', '--main_process_port', '0']
    benchmarks = []
    for repeat, order in ((1, ('control', 'reference', 'optimized')),
                          (2, ('optimized', 'reference', 'control'))):
        for case in order:
            name = f'benchmark-{case}-{repeat}'
            benchmarks.append(job(name, [*launch, str(source / 'scripts/benchmark_bdot.py'),
                '--config', str(root / f'benchmark-{case}.json'), '--output', '{run_dir}/'+name+'.json',
                '--device', 'cuda', '--batch-size', '16', '--accumulation', '4', '--length', '1024',
                '--document-length', '1024', '--warmup', '3', '--steps', '15', '--bf16',
                '--attention-backend', 'fa4', '--expert-backend', 'grouped'], name+'.json', status='completed'))
    checks = job('optimization_gate', ['{python}', str(source / 'scripts/stop_optimized_launch.py'),
                                      'gate', '--run', '{run_dir}'], 'optimization-gate.json')
    # All numerical, capacity, distributed and throughput checks precede any fit.
    jobs = [numerics, jobs[0], ddp, jobs[1], *benchmarks, checks, *jobs[2:]]
    # prepare() wrote the first manifest here; no process consumes it before launch.
    (root / 'jobs.json').unlink()
    write(root / 'jobs.json', dict(jobs=jobs))
    write(root / 'optimization.json', dict(status='prepared', ce_backend='native', shared_prefix=True,
          parent=str(parent), scope='Fresh matched D/control and optimized DualCE6, 10k adaptation updates each',
          training_gated_by='optimization-gate.json and eval_smoke/summary.json',
          previous_outputs='Preserved; no weights, optimizer state, caches or active source are overwritten.'))
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    g = sub.add_parser('gate'); g.add_argument('--run', type=Path, required=True)
    launch = sub.add_parser('launch')
    for key in ('root', 'parent', 'previous', 'smoke-previous'):
        launch.add_argument('--' + key, type=Path, required=True)
    launch.add_argument('--host', required=True); launch.add_argument('--session', required=True)
    a = p.parse_args()
    if a.action == 'gate':
        gate(a.run)
        return
    # Launch orchestration is not included in the immutable training snapshot.
    # The gate needs only its JSON receipts, not the launch-only import graph.
    from scripts.bdot_stop_launch import snapshot
    from scripts.stop_long_launch import OVERLAYS
    from scripts.native_baseline_pipeline import burn_gate
    from scripts.gpu_process import inspect

    if socket.gethostname() != a.host:
        raise ValueError('Wrong node')
    burn_gate(a.previous)
    original = a.previous / 'source'
    for rel, sha in read(original / 'SOURCE_MANIFEST.json')['files'].items():
        if hashlib.sha256((original / rel).read_bytes()).hexdigest() != sha:
            raise ValueError('Predecessor source changed: ' + rel)
    a.root.mkdir(parents=True, exist_ok=False)
    source = a.root / 'source'
    snapshot(original, source, (*OVERLAYS, *EXTRA))
    prepare_optimized(a.parent, a.smoke_previous, a.root, source)
    burn_gate(a.previous)
    write(a.root / 'preflight.json', inspect())
    subprocess.run([sys.executable, str(source / 'scripts/bdot_b200_launch.py'),
        '--root', str(a.root), '--host', a.host, '--session', a.session,
        '--jobs-config', str(a.root / 'jobs.json'),
        '--predecessor-burn-script', str(original / 'resources/llm_pretrain_burn.py')], cwd=source, check=True)


if __name__ == '__main__':
    main()
