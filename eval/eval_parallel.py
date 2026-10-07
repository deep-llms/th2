"""Launch eval_checkpoint.py on multiple checkpoints in parallel, one per GPU.

Uses a queue + worker pool: when any GPU finishes, it immediately picks up
the next checkpoint. No idle GPUs waiting for slow stragglers.

Usage:
  python eval/eval_parallel.py --checkpoints ckpt1 ckpt2 --output-dir eval_results \
      --bench-only --english-only --tasks hellaswag xnli --num-gpus 2
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
    if args.english_only:
        cmd.append('--english-only')
    for name in ('tokenizer_name', 'attention_backend', 'batch_size', 'num_fewshot',
                 'seed', 'limit', 'block_size', 'stride', 'dataset_root', 'dataset_manifest'):
        value = getattr(args, name)
        if value is not None:
            cmd += ['--' + name.replace('_', '-'), str(value)]
    for name in ('tasks', 'langs'):
        if getattr(args, name):
            cmd += ['--' + name, *getattr(args, name)]
    return cmd


def launch(script, ckpt, gpu_id, args):
    cmd = build_cmd(script, ckpt, args)
    output = args.destinations[ckpt]
    os.makedirs(output, exist_ok=False)
    cmd += ['--output-dir', output]
    log_path = os.path.join(output, "eval.log")
    log_file = open(log_path, "x")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    except BaseException:
        log_file.close()
        raise
    return {"process": p, "ckpt": ckpt, "gpu_id": gpu_id, "log_file": log_file,
            "log_path": log_path, "output": output}


def main():
    parser = argparse.ArgumentParser(description="Parallel evaluation across GPUs")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Checkpoint paths")
    parser.add_argument("--eval-dir", default=None,
                        help="Eval data directory")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
    parser.add_argument("--ppl-only", action="store_true", help="Only run perplexity")
    parser.add_argument("--bench-only", action="store_true", help="Only run benchmarks")
    parser.add_argument("--num-gpus", type=int, default=8, help="Number of GPUs available")
    parser.add_argument('--gpu-ids', nargs='+', help='Explicit GPU IDs/UUIDs; otherwise honor CUDA_VISIBLE_DEVICES')
    parser.add_argument('--output-dir', required=True, help='Fresh parent directory for evaluation outputs')
    parser.add_argument('--english-only', action='store_true')
    parser.add_argument('--tasks', nargs='+')
    parser.add_argument('--langs', nargs='+')
    parser.add_argument('--tokenizer-name')
    parser.add_argument('--attention-backend', choices=['sdpa', 'fa4'])
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-fewshot', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--dataset-root')
    parser.add_argument('--dataset-manifest')
    parser.add_argument('--block-size', type=int)
    parser.add_argument('--stride', type=int)
    parser.add_argument("--log", default="eval_parallel.log", help="Log file for parallel launcher output")
    args = parser.parse_args()
    if args.ppl_only and args.bench_only or not args.bench_only and args.eval_dir is None:
        parser.error('Choose a valid mode; --eval-dir is required for perplexity')
    if args.num_gpus < 1:
        parser.error('--num-gpus must be positive')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    gpu_ids = args.gpu_ids or (visible.split(',') if visible is not None else list(map(str, range(args.num_gpus))))
    if not gpu_ids or any(not g.strip() or g == '-1' for g in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
        parser.error('Select distinct visible GPUs')
    gpu_ids = gpu_ids[:args.num_gpus]
    checkpoints = [os.path.abspath(p) for p in args.checkpoints]
    if len(set(checkpoints)) != len(checkpoints) or any(not os.path.isdir(p) for p in checkpoints):
        parser.error('Every checkpoint must exist and be listed only once')
    if os.path.exists(args.output_dir):
        parser.error('--output-dir must be fresh')
    os.makedirs(args.output_dir)
    args.destinations = {p: os.path.join(args.output_dir, f'{i:02d}-{os.path.basename(os.path.dirname(p))}-{os.path.basename(p)}')
                         for i, p in enumerate(checkpoints)}
    with open(os.path.join(args.output_dir, 'checkpoints.json'), 'x') as f:
        json.dump(args.destinations, f, indent=2)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.log, mode="w"),
        ],
    )

    logger.info(f"Evaluating {len(checkpoints)} checkpoints across {len(gpu_ids)} GPUs")
    logger.info("")

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_checkpoint.py")
    queue = deque(checkpoints)
    free_gpus = deque(gpu_ids)
    active = []
    completed = []
    start_time = time.time()

    owned = []
    try:
        while queue or active:
            while queue and free_gpus:
                ckpt, gpu_id = queue.popleft(), free_gpus.popleft()
                logger.info(f"  START  GPU {gpu_id}: {ckpt}")
                job = launch(script, ckpt, gpu_id, args)
                owned.append(job)
                active.append(job)
            time.sleep(2)
            for job in active[:]:
                ret = job['process'].poll()
                if ret is None:
                    continue
                job['log_file'].close()
                status = 'OK' if ret == 0 else f'FAILED (code {ret})'
                logger.info(f"  DONE GPU {job['gpu_id']}: {status} - {job['ckpt']}")
                completed.append(job)
                active.remove(job)
                free_gpus.append(job['gpu_id'])
    finally:
        # Only direct child handles launched here, never process-name matching.
        for job in owned:
            process = job['process']
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            job['log_file'].close()

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
        ppl_path = os.path.join(job['output'], "eval_ppl.json")
        bench_path = os.path.join(job['output'], "eval_benchmarks.json")

        logger.info(f"\n[{status}] {name}:")
        if status != 'OK':
            continue

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
    return int(any(job['process'].returncode != 0 for job in completed))


if __name__ == "__main__":
    sys.exit(main())
