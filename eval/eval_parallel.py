"""Queue PPL + benchmarks for multiple checkpoints (English by default)."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.runtime import offline
offline()
from eval.runtime import checkpoint_specs, languages
from eval.benchmarks import DEFAULT_GROUPS, task_plan
from eval.parallel import add_arguments, common_args, run


def build_jobs(args):
    checkpoints = checkpoint_specs(args.checkpoints)
    selected = languages(args.languages)
    plan, _ = task_plan(selected, args.task_groups)
    if getattr(args, 'diagnostic_bundle', None):
        diagnostics = getattr(args, 'diagnostics', ['frequency', 'spectra', 'gradients'])
        if not diagnostics or len(set(diagnostics)) != len(diagnostics):
            raise ValueError('Select nonempty unique diagnostics')
    root = Path(args.output_dir).resolve()
    jobs = []
    for name, checkpoint in checkpoints.items():
        for stage in ('ppl', 'benchmarks'):
            output = root/name/f'{stage}.json'
            command = [sys.executable, '-u', '-m', 'eval.eval_checkpoint'] + common_args(args, checkpoint)
            command += ['--output', str(output)]
            if stage == 'ppl':
                command += ['--ppl-only', '--eval-dir', str(Path(args.eval_dir).resolve()),
                    '--batch-size', str(args.batch_size), '--preprocessing-num-workers', str(args.preprocessing_num_workers),
                    '--preprocessing-batch-size', str(args.preprocessing_batch_size)]
                if args.preprocessing_cache_dir:
                    command += ['--preprocessing-cache-dir', str(Path(args.preprocessing_cache_dir).resolve())]
            else:
                command += ['--bench-only', '--task-groups', *args.task_groups]
            jobs.append(dict(name=f'{name}_{stage}', argv=command, result=str(output),
                checkpoint=checkpoint, stage=stage, tasks=[item['task'] for item in plan],
                expected=dict(languages=selected)))
        bundle = getattr(args, 'diagnostic_bundle', None)
        if bundle:
            from capacity_allocation.data import sha256
            manifest_path = Path(bundle).resolve(strict=True)/'manifest.json'
            for diagnostic in getattr(args, 'diagnostics', ['frequency', 'spectra', 'gradients']):
                output = root/name/f'diagnostic_{diagnostic}.json'
                command = [sys.executable, '-u', '-m', 'eval.diagnostics_checkpoint',
                    '--checkpoint', checkpoint, '--diagnostic-bundle', str(manifest_path.parent),
                    '--languages', args.languages, '--diagnostics', diagnostic,
                    '--precision', args.precision, '--output', str(output)]
                if args.tokenizer_name:
                    command += ['--tokenizer-name', str(Path(args.tokenizer_name).resolve())]
                jobs.append(dict(name=f'{name}_diagnostic_{diagnostic}', argv=command,
                    result=str(output), checkpoint=checkpoint, stage='diagnostics',
                    tasks=[diagnostic], expected=dict(languages=selected,
                    diagnostic_manifest_sha256=sha256(manifest_path))))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument('--eval-dir', required=True)
    parser.add_argument('--task-groups', nargs='+', default=list(DEFAULT_GROUPS))
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--preprocessing-num-workers', type=int, default=160)
    parser.add_argument('--preprocessing-batch-size', type=int, default=1000)
    parser.add_argument('--preprocessing-cache-dir')
    parser.add_argument('--diagnostic-bundle', help='Optional frozen diagnostic bundle; no training changes')
    parser.add_argument('--diagnostics', nargs='+', choices=('frequency', 'spectra', 'gradients'),
                        default=['frequency', 'spectra', 'gradients'])
    args = parser.parse_args()
    jobs = build_jobs(args)
    if args.dry_run:
        print(json.dumps(jobs, indent=2))
    else:
        run(jobs, args.gpus, args.output_dir, Path(__file__).resolve().parents[1])


if __name__ == '__main__':
    main()
