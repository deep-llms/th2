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
    args = parser.parse_args()
    jobs = build_jobs(args)
    if args.dry_run:
        print(json.dumps(jobs, indent=2))
    else:
        run(jobs, args.gpus, args.output_dir, Path(__file__).resolve().parents[1])


if __name__ == '__main__':
    main()
