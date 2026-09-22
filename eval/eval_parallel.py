"""Launch eval_checkpoint.py on multiple checkpoints in parallel, one per GPU.

Uses a queue + worker pool: when any GPU finishes, it immediately picks up
the next checkpoint. No idle GPUs waiting for slow stragglers.

Usage:
  python eval/eval_parallel.py --eval-dir data/Qwen_Qwen3-0.6B/eval --bf16
  python eval/eval_parallel.py --eval-dir data/Qwen_Qwen3-0.6B/eval --bf16 --ppl-only
  python eval/eval_parallel.py --eval-dir data/Qwen_Qwen3-0.6B/eval --bf16 --bench-only
  python eval/eval_parallel.py --checkpoints ckpt1 ckpt2 ckpt3 --eval-dir ... --bf16
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from collections import deque

logger = logging.getLogger(__name__)


DEFAULT_OUTPUT_BASE = "/opt/dlami/nvme/smoke_test_outputs"

DEFAULT_CHECKPOINTS = [
    f"{DEFAULT_OUTPUT_BASE}/baseline/checkpoint-6500",
    f"{DEFAULT_OUTPUT_BASE}/S3_a015/checkpoint-6500",
    f"{DEFAULT_OUTPUT_BASE}/S3_a02/checkpoint-6500",
]


def build_cmd(script, ckpt, args):
    cmd = [sys.executable, script, "--checkpoint", ckpt, "--device", "cuda"]
    if not args.bench_only:
        cmd += ["--eval-dir", args.eval_dir]
    if args.bf16:
        cmd.append("--bf16")
    if args.ppl_only:
        cmd.append("--ppl-only")
    if args.bench_only:
        cmd.append("--bench-only")
    return cmd


def launch(script, ckpt, gpu_id, args):
    cmd = build_cmd(script, ckpt, args)
    log_path = os.path.join(ckpt, "eval.log")
    log_file = open(log_path, "w")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    return {"process": p, "ckpt": ckpt, "gpu_id": gpu_id, "log_file": log_file, "log_path": log_path}


def main():
    parser = argparse.ArgumentParser(description="Parallel evaluation across GPUs")
    parser.add_argument("--checkpoints", nargs="+", default=None,
                        help="Checkpoint paths")
    parser.add_argument("--eval-dir", default=None,
                        help="Eval data directory")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
    parser.add_argument("--ppl-only", action="store_true", help="Only run perplexity")
    parser.add_argument("--bench-only", action="store_true", help="Only run benchmarks")
    parser.add_argument("--num-gpus", type=int, default=8, help="Number of GPUs available")
    parser.add_argument("--log", default="eval_parallel.log", help="Log file for parallel launcher output")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.log, mode="w"),
        ],
    )

    checkpoints = args.checkpoints or DEFAULT_CHECKPOINTS

    valid = []
    for ckpt in checkpoints:
        if os.path.isdir(ckpt):
            valid.append(ckpt)
        else:
            logger.info(f"SKIPPING: {ckpt} (not found)")
    checkpoints = valid

    if not checkpoints:
        logger.info("No valid checkpoints found.")
        return

    logger.info(f"Evaluating {len(checkpoints)} checkpoints across {args.num_gpus} GPUs")
    logger.info("")

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_checkpoint.py")
    queue = deque(checkpoints)
    free_gpus = deque(range(args.num_gpus))
    active = []
    completed = []
    start_time = time.time()

    # Fill initial slots
    while queue and free_gpus:
        ckpt = queue.popleft()
        gpu_id = free_gpus.popleft()
        logger.info(f"  START  GPU {gpu_id}: {ckpt}")
        job = launch(script, ckpt, gpu_id, args)
        active.append(job)

    # Poll until all done
    while active:
        time.sleep(2)

        still_active = []
        for job in active:
            ret = job["process"].poll()
            if ret is not None:
                job["log_file"].close()
                elapsed = time.time() - start_time
                status = "OK" if ret == 0 else f"FAILED (code {ret})"
                logger.info(f"  DONE   GPU {job['gpu_id']}: {status} - {job['ckpt']}  [{elapsed:.0f}s elapsed]")
                if ret != 0:
                    logger.info(f"         See log: {job['log_path']}")

                completed.append(job)
                free_gpus.append(job["gpu_id"])

                # Launch next from queue (small delay for GPU cleanup)
                if queue:
                    time.sleep(10)
                    ckpt = queue.popleft()
                    gpu_id = free_gpus.popleft()
                    logger.info(f"  START  GPU {gpu_id}: {ckpt}")
                    new_job = launch(script, ckpt, gpu_id, args)
                    still_active.append(new_job)
            else:
                still_active.append(job)

        active = still_active

    total_elapsed = time.time() - start_time
    logger.info(f"\nAll {len(completed)} evaluations done in {total_elapsed:.0f}s")

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    for job in completed:
        ckpt = job["ckpt"]
        name = os.path.basename(os.path.dirname(ckpt)) + "/" + os.path.basename(ckpt)
        status = "OK" if job["process"].returncode == 0 else "FAILED"
        ppl_path = os.path.join(ckpt, "eval_ppl.json")
        bench_path = os.path.join(ckpt, "eval_benchmarks.json")

        logger.info(f"\n[{status}] {name}:")

        if os.path.isfile(ppl_path):
            with open(ppl_path) as f:
                ppl = json.load(f)
            for lang, r in sorted(ppl.items()):
                logger.info(f"  PPL   {lang:<5} {r['perplexity']:>10.2f}  (loss={r['loss']:.4f})")

        if os.path.isfile(bench_path):
            with open(bench_path) as f:
                bench = json.load(f)
            for task, r in sorted(bench.items()):
                acc = r.get("acc,none", r.get("acc", "?"))
                if isinstance(acc, float):
                    logger.info(f"  BENCH {task:<30} acc={acc:.4f}")


if __name__ == "__main__":
    main()
