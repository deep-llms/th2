"""Bounded before/after dispatch verification using an immutable old package."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--accelerate-config', type=Path, required=True)
    a = p.parse_args()
    if not (a.reference/'bdot/model.py').is_file():
        p.error('Missing frozen reference package')
    a.output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', checks={}, benchmarks={}, profiles={})

    def save():
        (a.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')

    def run(name, argv, gpus='0'):
        print('DISPATCH_START '+name, flush=True)
        with (a.output/(name+'.log')).open('x') as log:
            subprocess.run([sys.executable, *map(str, argv)], cwd=ROOT,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': gpus},
                stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1200)

    def bench(name, cfg, backend, progress, source=None, profile=False):
        path = a.output/(name+'.json')
        argv = ['-m', 'accelerate.commands.launch', '--config_file', a.accelerate_config,
            '--main_process_port', '0', 'scripts/benchmark_bdot.py',
            '--config', ROOT/'configs/bdot'/f'{cfg}.json', '--output', path,
            '--device', 'cuda', '--batch-size', '16', '--accumulation', '4', '--length', '1024',
            '--warmup', '5', '--steps', '1' if profile else '50', '--bf16',
            '--attention-backend', 'fa4', '--progress', str(progress), '--expert-backend', backend]
        if cfg == 'P1-B':
            argv += ['--router-prior', '0']
        if source:
            argv += ['--bdot-source', source]
        if profile:
            argv += ['--profile']
        run(name, argv, '0,1,2,3,4,5,6,7')
        report['profiles' if profile else 'benchmarks'][name] = json.loads(path.read_text())
        save()

    try:
        run('parity', ['tests/bdot/dispatch_probe.py', '--device', 'cuda',
                       '--reference', a.reference, '--output', a.output/'parity.json'])
        report['checks']['parity'] = json.loads((a.output/'parity.json').read_text())
        save()
        for label, flags in [('home', ['--home-only']), ('soft', ['--soft']), ('quota', ['--no-reuse'])]:
            path = a.output/('ddp_'+label)
            run('ddp_'+label, ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                'tests/bdot/ddp_probe.py', path, '--device', 'cuda', '--bf16', '--checkpointing',
                '--expert-backend', 'grouped', *flags], '0,1')
            report['checks']['ddp_'+label] = [json.loads((path/f'rank{r}.json').read_text()) for r in range(2)]
            save()
        for curriculum in ('A', 'B'):
            path = a.output/('resume_'+curriculum)
            run('resume_'+curriculum, ['tests/bdot/resume_gpu_probe.py', 'suite', '--output', path,
                '--accelerate-config', a.accelerate_config, '--steps', '120', '--split', '60',
                '--soft-split', '10', '--expert-backend', 'grouped', '--curriculum', curriculum],
                '0,1,2,3,4,5,6,7')
            report['checks']['resume_'+curriculum] = json.loads((path/'summary.json').read_text())
            save()
        cases = [('D', 'D', 'reference', 1.), ('hard', 'P1-B', 'grouped', 1.),
                 ('soft', 'P1-B', 'grouped', .01)]
        for repeat in range(2):
            bench(f'native_r{repeat}', 'dense_native', 'reference', 1.)
            for label, cfg, backend, progress in cases:
                versions = [('old', a.reference), ('new', None)]
                if repeat:
                    versions.reverse()
                for version, source in versions:
                    bench(f'{label}_{version}_r{repeat}', cfg, backend, progress, source)
                if label != 'D':
                    # Also measure the default backend: combining scatter copies
                    # changes its cost, so the old grouped ranking may not hold.
                    bench(f'{label}_reference_r{repeat}', cfg, 'reference', progress)
        for label, cfg, backend, progress in cases:
            bench(f'profile_{label}', cfg, backend, progress, profile=True)
        report['comparison'] = {
            label: {version: statistics.mean(report['benchmarks'][f'{label}_{version}_r{i}']['seconds_per_update']
                                             for i in range(2)) for version in ('old', 'new')}
            for label, *_ in cases}
        report['status'] = 'passed'
        save()
        print('DISPATCH_VERIFICATION_PASSED', flush=True)
    except Exception as error:
        report.update(status='failed', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
