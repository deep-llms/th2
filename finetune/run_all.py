"""Queue independent checkpoint/task/seed fine-tunes; train/eval English by default."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.runtime import offline
offline()
from eval.runtime import checkpoint_specs, languages
from eval.benchmarks import task_plan
from eval.parallel import add_arguments, common_args, run
from finetune.tasks import TASK_CONFIGS


def build_jobs(args):
    checkpoints = checkpoint_specs(args.checkpoints)
    selected = languages(args.languages)
    if len(languages(args.train_language)) != 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('One training language and unique seeds required')
    if not args.tasks or len(set(args.tasks)) != len(args.tasks) or set(args.tasks)-set(TASK_CONFIGS):
        raise ValueError('Choose unique supported fine-tuning tasks')
    root = Path(args.output_dir).resolve()
    jobs = []
    for task in args.tasks:
        plan, _ = task_plan(selected, [task])
        task_plan(args.train_language, [task])
        for name, checkpoint in checkpoints.items():
            for seed in args.seeds:
                jobname = f'{name}_{task}_{args.train_language}_seed{seed}'
                output = root/jobname
                command = [sys.executable, '-u', '-m', 'finetune.train'] + common_args(args, checkpoint)
                command += ['--task', task, '--train-language', args.train_language,
                            '--seed', str(seed), '--output-dir', str(output)]
                jobs.append(dict(name=jobname, argv=command, result=str(output/'result.json'),
                    checkpoint=checkpoint, stage='finetune', tasks=[item['task'] for item in plan],
                    expected=dict(languages=selected, train_language=args.train_language, task=task, seed=seed)))
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument('--tasks', nargs='+', choices=TASK_CONFIGS, default=list(TASK_CONFIGS))
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    parser.add_argument('--train-language', default='en')
    args = parser.parse_args()
    jobs = build_jobs(args)
    if args.dry_run:
        print(json.dumps(jobs, indent=2))
    else:
        run(jobs, args.gpus, args.output_dir, Path(__file__).resolve().parents[1])


if __name__ == '__main__':
    main()
