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
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def load_model(checkpoint_path, device, dtype=None):
    config = AutoConfig.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, config=config, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--eval-dir", default=None, help="Eval data directory (required for PPL)")
    parser.add_argument("--tokenizer-name", default=None, help="Tokenizer (default: from checkpoint)")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")

    # PPL args
    parser.add_argument("--block-size", type=int, default=2048, help="PPL context window size")
    parser.add_argument("--stride", type=int, default=None, help="PPL sliding window stride")
    parser.add_argument("--langs", nargs="+", default=None, help="Languages to evaluate")

    # Benchmark args
    parser.add_argument("--tasks", nargs="+", default=None, help="Benchmark groups (default: all)")
    parser.add_argument("--num-fewshot", type=int, default=0, help="Few-shot examples")
    parser.add_argument("--batch-size", type=int, default=16, help="Benchmark batch size")

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

    # Load model once
    dtype = torch.bfloat16 if args.bf16 else None
    tokenizer_name = args.tokenizer_name or args.checkpoint
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, args.device, dtype=dtype)
    print(f"  Device:     {args.device}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    # --- Perplexity ---
    if run_ppl:
        print("\n" + "-" * 60)
        print("  PERPLEXITY EVALUATION")
        print("-" * 60)

        from eval.ppl import eval_ppl, print_ppl_results
        ppl_results = eval_ppl(
            model, tokenizer, args.eval_dir,
            block_size=args.block_size,
            stride=args.stride,
            device=args.device,
            langs=args.langs,
        )
        print_ppl_results(ppl_results)
        all_results["perplexity"] = ppl_results

        with open(os.path.join(args.output_dir, "eval_ppl.json"), "w") as f:
            json.dump(ppl_results, f, indent=2)

    # --- Benchmarks ---
    if run_bench:
        print("\n" + "-" * 60)
        print("  BENCHMARK EVALUATION")
        print("-" * 60)

        from eval.benchmarks import eval_benchmarks, print_benchmark_results, TASK_CONFIGS
        task_groups = args.tasks or list(TASK_CONFIGS.keys())
        bench_results = eval_benchmarks(
            model, tokenizer,
            task_groups=task_groups,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            device=args.device,
        )
        print_benchmark_results(bench_results)
        all_results["benchmarks"] = bench_results["results"]

        with open(os.path.join(args.output_dir, "eval_benchmarks.json"), "w") as f:
            json.dump(bench_results["results"], f, indent=2, default=str)

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
