"""Run the full fine-tuning benchmark battery.

Supports two execution modes:
  --num-gpus 1  (default): sequential, one run at a time (original behavior)
  --num-gpus N: parallel, N runs at once on N GPUs via subprocess + CUDA_VISIBLE_DEVICES

Each (task, mode, arm, seed) is one run. Parallel mode launches each run as a
subprocess of finetune/train.py pinned to a free GPU — same eval_parallel.py
queue pattern used throughout the project.

Skip-if-exists: completed runs (JSON exists) are skipped on restart.

Usage:
  python finetune/run_all.py \
      --checkpoints baseline=/path/checkpoint-10000 lowrank=/path/checkpoint-10000 \
      --tasks ag_news sst2 xnli paws_x hellaswag \
      --modes full probe \
      --seeds 42 123 456 \
      --num-gpus 8 \
      --output-dir /opt/dlami/nvme/sparse_emb_outputs/finetune
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finetune.tasks import TASK_CONFIGS


def detect_gpus():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True)
        return [int(x.strip()) for x in r.stdout.strip().split("\n")
                if x.strip()]
    except Exception:
        return [0]


def build_summary(results, output_dir):
    """Build a markdown summary table from all result JSONs."""
    by_task_mode = {}
    for r in results:
        if "error" in r:
            continue
        key = (r["task"], r["mode"])
        arm = r.get("arm_name", os.path.basename(os.path.dirname(
            os.path.normpath(r["checkpoint"]))))
        by_task_mode.setdefault(key, {}).setdefault(arm, []).append(r)

    lines = ["# Fine-tune Benchmark Results\n"]
    all_arms = sorted({arm for arms in by_task_mode.values()
                       for arm in arms})

    for (task, mode), arms_data in sorted(by_task_mode.items()):
        lines.append(f"\n## {task} — {mode}\n")
        test_keys = sorted({tk for runs in arms_data.values()
                            for r in runs for tk in r["test_results"]})
        for tk in test_keys:
            header = f"| {tk} | " + " | ".join(all_arms) + " |"
            lines.append(header)
            lines.append("|" + "---|" * (len(all_arms) + 1))
            row = [tk]
            for arm in all_arms:
                runs = arms_data.get(arm, [])
                vals = [r["test_results"].get(tk, 0.0) for r in runs]
                if vals:
                    import numpy as np
                    mean = np.mean(vals)
                    std = np.std(vals)
                    row.append(f"{mean:.4f}±{std:.4f}" if len(vals) > 1
                               else f"{vals[0]:.4f}")
                else:
                    row.append("-")
            lines.append("| " + " | ".join(row) + " |")

    report = "\n".join(lines)
    path = os.path.join(output_dir, "summary.md")
    with open(path, "w") as f:
        f.write(report + "\n")
    print(f"\nSummary: {path}")
    print(report)
    return report


def run_sequential(jobs, args):
    """Original sequential mode: one run at a time."""
    from finetune.train import train_one
    all_results = []
    for i, (task, mode, arm_name, ckpt_path, seed) in enumerate(jobs):
        tag = f"{task}/{mode}/{arm_name}/seed{seed}"
        print(f"\n{'='*60}")
        print(f"  [{i+1}/{len(jobs)}] {tag}")
        print(f"{'='*60}")

        out_json = os.path.join(
            args.output_dir, f"{task}_{mode}_{arm_name}_seed{seed}.json")
        if os.path.isfile(out_json):
            print(f"  SKIP (already exists)")
            with open(out_json) as f:
                all_results.append(json.load(f))
            continue

        try:
            result = train_one(ckpt_path, task, mode, seed,
                               args.device, args.output_dir,
                               args.tokenizer_name, args.num_workers)
            all_results.append(result)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            all_results.append({"checkpoint": ckpt_path, "task": task,
                                "mode": mode, "seed": seed, "arm_name": arm_name,
                                "error": str(e), "test_results": {}})
        time.sleep(10)
    return all_results


def launch_one(task, mode, arm_name, ckpt_path, seed, gpu_id, args):
    """Launch one finetune/train.py as a subprocess on a specific GPU."""
    out_json = os.path.join(
        args.output_dir, f"{task}_{mode}_{arm_name}_seed{seed}.json")
    log_path = os.path.join(
        args.output_dir, f"{task}_{mode}_{arm_name}_seed{seed}.log")
    log_file = open(log_path, "w")

    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "train.py"),
           "--checkpoint", ckpt_path,
           "--task", task, "--mode", mode, "--seed", str(seed),
           "--output-dir", args.output_dir,
           "--tokenizer-name", args.tokenizer_name,
           "--device", "cuda"]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["FINETUNE_ARM_NAME"] = arm_name

    p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                         env=env)
    return {"process": p, "task": task, "mode": mode, "arm": arm_name,
            "seed": seed, "gpu_id": gpu_id, "log_file": log_file,
            "log_path": log_path, "out_json": out_json,
            "checkpoint": ckpt_path}


def run_parallel(jobs, args, gpu_ids):
    """Parallel mode: queue of jobs across N GPUs."""
    queue = deque()
    skipped = []
    for task, mode, arm_name, ckpt_path, seed in jobs:
        out_json = os.path.join(
            args.output_dir, f"{task}_{mode}_{arm_name}_seed{seed}.json")
        if os.path.isfile(out_json):
            skipped.append(out_json)
        else:
            queue.append((task, mode, arm_name, ckpt_path, seed))

    if skipped:
        print(f"Skipping {len(skipped)} already-completed runs")
    print(f"Running {len(queue)} jobs across {len(gpu_ids)} GPUs")

    free_gpus = deque(gpu_ids)
    active = []
    completed = []
    failed = []
    t_start = time.time()

    while queue and free_gpus:
        task, mode, arm_name, ckpt_path, seed = queue.popleft()
        gpu_id = free_gpus.popleft()
        tag = f"{task}/{mode}/{arm_name}/s{seed}"
        print(f"  START GPU {gpu_id}: {tag}")
        job = launch_one(task, mode, arm_name, ckpt_path, seed, gpu_id, args)
        active.append(job)

    while active:
        time.sleep(5)
        still_active = []
        for job in active:
            ret = job["process"].poll()
            if ret is None:
                still_active.append(job)
                continue
            job["log_file"].close()
            tag = f"{job['task']}/{job['mode']}/{job['arm']}/s{job['seed']}"
            elapsed = time.time() - t_start

            if ret == 0 and os.path.isfile(job["out_json"]):
                status = "OK"
                completed.append(job)
            else:
                status = f"FAILED (code {ret})"
                failed.append(job)

            print(f"  DONE  GPU {job['gpu_id']}: {status} — {tag}  "
                  f"[{elapsed:.0f}s]")
            if "FAILED" in status:
                print(f"        see {job['log_path']}")

            free_gpus.append(job["gpu_id"])

            if queue:
                time.sleep(5)
                task, mode, arm_name, ckpt_path, seed = queue.popleft()
                gpu_id = free_gpus.popleft()
                tag2 = f"{task}/{mode}/{arm_name}/s{seed}"
                print(f"  START GPU {gpu_id}: {tag2}")
                new_job = launch_one(task, mode, arm_name, ckpt_path, seed,
                                     gpu_id, args)
                still_active.append(new_job)

        active = still_active

    total_elapsed = time.time() - t_start
    print(f"\nAll done in {total_elapsed/60:.1f}min "
          f"({len(completed)} OK, {len(failed)} failed, "
          f"{len(skipped)} skipped)")

    # Load all results (completed + skipped)
    all_results = []
    for task, mode, arm_name, ckpt_path, seed in jobs:
        out_json = os.path.join(
            args.output_dir, f"{task}_{mode}_{arm_name}_seed{seed}.json")
        if os.path.isfile(out_json):
            with open(out_json) as f:
                all_results.append(json.load(f))
        else:
            all_results.append({"checkpoint": ckpt_path, "task": task,
                                "mode": mode, "seed": seed, "arm_name": arm_name,
                                "error": "no output", "test_results": {}})
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run fine-tune benchmark battery")
    parser.add_argument(
        "--checkpoints", nargs="+", required=True,
        help="arm=path pairs, e.g. baseline=/path/to/checkpoint-10000")
    parser.add_argument("--tasks", nargs="+",
                        default=list(TASK_CONFIGS.keys()))
    parser.add_argument("--modes", nargs="+", default=["full", "probe"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="temp/finetune")
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="GPUs for parallel mode (default: auto-detect; "
                             "1 = sequential)")
    args = parser.parse_args()

    arm_paths = {}
    for spec in args.checkpoints:
        if "=" in spec:
            name, path = spec.split("=", 1)
        else:
            name = os.path.basename(os.path.dirname(
                os.path.normpath(spec)))
            path = spec
        arm_paths[name] = path

    os.makedirs(args.output_dir, exist_ok=True)

    # Build job list
    jobs = []
    for task in args.tasks:
        for mode in args.modes:
            for arm_name, ckpt_path in arm_paths.items():
                for seed in args.seeds:
                    jobs.append((task, mode, arm_name, ckpt_path, seed))

    print(f"Total jobs: {len(jobs)} "
          f"({len(args.tasks)} tasks × {len(args.modes)} modes × "
          f"{len(arm_paths)} arms × {len(args.seeds)} seeds)")

    if args.num_gpus is None:
        gpu_ids = detect_gpus()
        args.num_gpus = len(gpu_ids)
        print(f"Auto-detected {args.num_gpus} GPUs")
    else:
        gpu_ids = list(range(args.num_gpus))

    t_start = time.time()
    if args.num_gpus <= 1:
        all_results = run_sequential(jobs, args)
    else:
        all_results = run_parallel(jobs, args, gpu_ids)

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed/3600:.1f}h")
    build_summary([r for r in all_results if "error" not in r], args.output_dir)


if __name__ == "__main__":
    main()
