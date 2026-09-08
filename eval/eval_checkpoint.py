"""Evaluate a local Stagewise/stock Qwen checkpoint; no downloads or GPU management."""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.runtime import offline
offline()
from eval.runtime import checkpoint_identity, languages, load_checkpoint
from eval.benchmarks import DEFAULT_GROUPS, task_plan, load_tasks, summarize_benchmarks, evaluate as benchmark_eval
from eval.ppl import evaluate as ppl_eval
from capacity_allocation.data import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--tokenizer-name', help='Local override; defaults to checkpoint tokenizer')
    parser.add_argument('--languages', default='en')
    parser.add_argument('--eval-dir')
    parser.add_argument('--dataset-root', help='Local benchmark snapshots in preserved org/repo layout')
    parser.add_argument('--task-groups', nargs='+', default=list(DEFAULT_GROUPS))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--ppl-only', action='store_true')
    mode.add_argument('--bench-only', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--precision', choices=('fp32', 'bf16'), default='bf16')
    parser.add_argument('--batch-size', type=int, default=1, help='PPL batch size')
    parser.add_argument('--benchmark-batch-size', type=int, default=8)
    parser.add_argument('--block-size', type=int, default=2048)
    parser.add_argument('--preprocessing-num-workers', type=int, default=160)
    parser.add_argument('--preprocessing-batch-size', type=int, default=1000)
    parser.add_argument('--preprocessing-cache-dir')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or (not args.bench_only and not args.eval_dir) or (
            not args.ppl_only and not args.dataset_root):
        parser.error('Fresh output and selected stage data paths are required')
    selected = languages(args.languages)
    plan, unavailable = task_plan(selected, args.task_groups) if not args.ppl_only else ([], [])
    tasks = load_tasks(plan, args.dataset_root) if plan else {}
    model, tokenizer = load_checkpoint(args.checkpoint, args.tokenizer_name, args.device, args.precision)
    result = dict(success=True, checkpoint=checkpoint_identity(args.checkpoint),
                  languages=selected, settings=vars(args), benchmark_plan=plan,
                  unavailable_benchmarks=unavailable)
    if not args.bench_only:
        result['ppl'] = ppl_eval(model, tokenizer, args.eval_dir, selected, device=args.device,
            precision=args.precision, batch_size=args.batch_size, block_size=args.block_size,
            workers=args.preprocessing_num_workers, map_batch_size=args.preprocessing_batch_size,
            cache_dir=args.preprocessing_cache_dir)
    if tasks:
        result['benchmarks'] = benchmark_eval(model, tokenizer, tasks, device=args.device,
            precision=args.precision, batch_size=args.benchmark_batch_size)
        result['benchmark_summaries'] = summarize_benchmarks(result['benchmarks'])
        result['lm_eval_version'] = '0.4.10'
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(f'EVALUATION_COMPLETE {output}', flush=True)


if __name__ == '__main__':
    main()
