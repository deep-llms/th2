"""Evaluate a single checkpoint: perplexity and/or benchmarks.

Loads the model once, runs both evaluations on the same GPU.

Usage:
  python eval/eval_checkpoint.py --checkpoint path/to/ckpt --eval-dir data/Qwen_Qwen3-0.6B/eval --bf16
  python eval/eval_checkpoint.py --checkpoint path/to/ckpt --eval-dir data/Qwen_Qwen3-0.6B/eval --bf16 --ppl-only
  python eval/eval_checkpoint.py --checkpoint path/to/ckpt --bf16 --bench-only
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer
from eval.models import load_checkpoint


def json_default(value):
    """Keep numerical scores numeric; string conversion is for config objects."""
    import numpy as np
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--eval-dir", default=None, help="Eval data directory (required for PPL)")
    parser.add_argument("--tokenizer-name", default=None, help="Tokenizer (default: from checkpoint)")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
    parser.add_argument('--attention-backend', choices=['sdpa', 'fa4'], default=None,
                        help='Default: saved backend; explicit overrides are recorded')

    # PPL args
    parser.add_argument("--block-size", type=int, default=None, help="PPL window (default: trained limit, at most 2048)")
    parser.add_argument("--stride", type=int, default=None, help="PPL sliding window stride")
    parser.add_argument("--langs", nargs="+", default=None, help="Languages to evaluate")

    # Benchmark args
    parser.add_argument("--tasks", nargs="+", default=None, help="Benchmark groups (default: all)")
    parser.add_argument("--num-fewshot", type=int, default=0, help="Few-shot examples")
    parser.add_argument("--batch-size", type=int, default=16, help="Benchmark batch size")
    parser.add_argument('--english-only', action='store_true', help='Select only English subsets')
    parser.add_argument('--seed', type=int, default=42, help='All evaluation/few-shot RNG seeds')
    parser.add_argument('--limit', type=int, default=None, help='Diagnostic examples per benchmark; omit for full results')
    parser.add_argument('--dataset-root', help='Local raw benchmark snapshots')
    parser.add_argument('--dataset-manifest', help='Pinned file hashes and task-to-snapshot mapping')

    # Mode
    parser.add_argument("--ppl-only", action="store_true", help="Only run perplexity")
    parser.add_argument("--bench-only", action="store_true", help="Only run benchmarks")

    parser.add_argument("--output-dir", default=None, help="Output directory (default: checkpoint dir)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.checkpoint

    if args.ppl_only and args.bench_only:
        parser.error("Cannot use --ppl-only and --bench-only together")

    run_ppl = not args.bench_only
    run_bench = not args.ppl_only

    if run_ppl and args.eval_dir is None:
        parser.error("--eval-dir is required for perplexity evaluation")
    if args.batch_size < 1 or args.num_fewshot < 0 or args.limit is not None and args.limit < 1:
        parser.error('Batch size/limit must be positive and num-fewshot nonnegative')
    if args.block_size is not None and args.block_size < 2:
        parser.error('--block-size must be at least two')
    dataset_paths = None
    if bool(args.dataset_root) != bool(args.dataset_manifest):
        parser.error('--dataset-root and --dataset-manifest must be supplied together')
    if run_bench:
        from eval.benchmarks import resolve_tasks, local_dataset_paths, task_configs
        names = resolve_tasks(args.tasks, args.english_only)
        if args.dataset_root:
            dataset_paths = local_dataset_paths(args.dataset_root, args.dataset_manifest)
            task_configs(names, dataset_paths)  # Fail before loading expensive weights.
    outputs = ['eval_metadata.json']
    if run_ppl:
        outputs.append('eval_ppl.json')
    if run_bench:
        outputs.extend(['eval_benchmarks.json', 'eval_benchmarks_full.json', 'eval_samples.jsonl'])
    if any(os.path.exists(os.path.join(args.output_dir, name)) for name in outputs):
        parser.error('Evaluation output already exists; choose a fresh --output-dir')

    # Load model once
    dtype = torch.bfloat16 if args.bf16 else None
    model, source, metadata = load_checkpoint(args.checkpoint, args.device, dtype, args.attention_backend)
    tokenizer_name = args.tokenizer_name or source
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device:     {args.device}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    metadata.update(arguments=vars(args), tokenizer_source=tokenizer_name, status='started')
    if dataset_paths is not None:
        from eval.models import file_hash
        metadata['dataset_manifest_sha256'] = file_hash(args.dataset_manifest)
        metadata['dataset_paths'] = dataset_paths
    from importlib.metadata import version
    metadata['packages'] = {p: version(p) for p in ('torch', 'transformers', 'datasets', 'accelerate')}
    if run_bench:
        metadata['packages']['lm_eval'] = version('lm_eval')
    def save_metadata():
        with open(os.path.join(args.output_dir, 'eval_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    save_metadata()

    # --- Perplexity ---
    if run_ppl:
        print("\n" + "-" * 60)
        print("  PERPLEXITY EVALUATION")
        print("-" * 60)

        from eval.ppl import eval_ppl, print_ppl_results
        ppl_results = eval_ppl(
            model, tokenizer, args.eval_dir,
            block_size=args.block_size or min(2048, model.config.max_position_embeddings),
            stride=args.stride,
            device=args.device,
            langs=args.langs,
        )
        print_ppl_results(ppl_results)

        with open(os.path.join(args.output_dir, "eval_ppl.json"), "w") as f:
            json.dump(ppl_results, f, indent=2)

    # --- Benchmarks ---
    if run_bench:
        print("\n" + "-" * 60)
        print("  BENCHMARK EVALUATION")
        print("-" * 60)

        from eval.benchmarks import eval_benchmarks, print_benchmark_results
        bench_results = eval_benchmarks(
            model, tokenizer,
            task_groups=args.tasks,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            device=args.device,
            english_only=args.english_only, seed=args.seed, limit=args.limit,
            dataset_paths=dataset_paths,
        )
        print_benchmark_results(bench_results)

        with open(os.path.join(args.output_dir, "eval_benchmarks.json"), "w") as f:
            json.dump(bench_results["results"], f, indent=2, default=json_default, allow_nan=False)
        with open(os.path.join(args.output_dir, 'eval_benchmarks_full.json'), 'w') as f:
            json.dump({k: v for k, v in bench_results.items() if k != 'samples'}, f,
                      indent=2, default=json_default, allow_nan=False)
        with open(os.path.join(args.output_dir, 'eval_samples.jsonl'), 'w') as f:
            for task, samples in bench_results.get('samples', {}).items():
                for sample in samples:
                    f.write(json.dumps(dict(task=task, **sample), default=json_default, allow_nan=False) + '\n')
    metadata['status'] = 'completed'
    save_metadata()

    # --- Summary ---
    print("\n" + "=" * 60)
    print(f"  RESULTS SAVED")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Model:      {model.config.architectures}")
    if run_ppl:
        print(f"  PPL:        {args.output_dir}/eval_ppl.json")
    if run_bench:
        print(f"  Benchmarks: {args.output_dir}/eval_benchmarks.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
