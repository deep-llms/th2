"""Run the full fine-tuning benchmark battery (generative approach).

Each (task, arm, seed) is one run: fine-tune with causal LM loss on
completion tokens, then evaluate with lm-evaluation-harness in-memory.

Parallel mode: N runs at once on N GPUs via subprocess + CUDA_VISIBLE_DEVICES.
Skip-if-exists: completed runs (JSON exists) are skipped on restart.

Usage:
  python finetune/run_all.py \
      --checkpoints baseline=/path/checkpoint-10000 lowrank=/path/checkpoint-10000 \
      --tasks hellaswag arc_easy piqa winogrande \
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
    import numpy as np

    by_task = {}
    for r in results:
        if "error" in r:
            continue
        task = r["task"]
        arm = r.get("arm_name", os.path.basename(os.path.dirname(
            os.path.normpath(r["checkpoint"]))))
        by_task.setdefault(task, {}).setdefault(arm, []).append(r)

    lines = ["# Fine-tune Benchmark Results (Generative)\n"]
    all_arms = sorted({arm for arms_data in by_task.values()
                       for arm in arms_data})

    for task, arms_data in sorted(by_task.items()):
        lines.append(f"\n## {task}\n")
        eval_keys = sorted({ek for runs in arms_data.values()
                            for r in runs for ek in r["eval_results"]})
        for ek in eval_keys:
            for metric in ["acc", "acc_norm"]:
                vals_by_arm = {}
                for arm in all_arms:
                    runs = arms_data.get(arm, [])
                    vals = [r["eval_results"].get(ek, {}).get(metric)
                            for r in runs]
                    vals = [v for v in vals if v is not None]
                    if vals:
                        vals_by_arm[arm] = vals

                if not vals_by_arm:
                    continue

                header = f"| {ek} ({metric}) | " + " | ".join(all_arms) + " |"
                lines.append(header)
                lines.append("|" + "---|" * (len(all_arms) + 1))
                row = [f"{ek} ({metric})"]
                for arm in all_arms:
                    vals = vals_by_arm.get(arm, [])
                    if vals:
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


def launch_one(task, arm_name, ckpt_path, seed, gpu_id, args):
    """Launch one finetune/train.py as a subprocess on a specific GPU."""
    out_json = os.path.join(
        args.output_dir, f"{task}_{arm_name}_seed{seed}.json")
    log_path = os.path.join(
        args.output_dir, f"{task}_{arm_name}_seed{seed}.log")
    log_file = open(log_path, "w")

    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "train.py"),
           "--checkpoint", ckpt_path,
           "--task", task, "--seed", str(seed),
           "--output-dir", args.output_dir,
           "--tokenizer-name", args.tokenizer_name,
           "--device", "cuda"]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["FINETUNE_ARM_NAME"] = arm_name

    p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                         env=env)
    return {"process": p, "task": task, "arm": arm_name,
            "seed": seed, "gpu_id": gpu_id, "log_file": log_file,
            "log_path": log_path, "out_json": out_json,
            "checkpoint": ckpt_path}


def run_parallel(jobs, args, gpu_ids):
    """Parallel mode: queue of jobs across N GPUs."""
    queue = deque()
    skipped = []
    for task, arm_name, ckpt_path, seed in jobs:
        out_json = os.path.join(
            args.output_dir, f"{task}_{arm_name}_seed{seed}.json")
        if os.path.isfile(out_json):
            skipped.append(out_json)
        else:
            queue.append((task, arm_name, ckpt_path, seed))

    if skipped:
        print(f"Skipping {len(skipped)} already-completed runs")
    print(f"Running {len(queue)} jobs across {len(gpu_ids)} GPUs")

    free_gpus = deque(gpu_ids)
    active = []
    completed = []
    failed = []
    t_start = time.time()

    while queue and free_gpus:
        task, arm_name, ckpt_path, seed = queue.popleft()
        gpu_id = free_gpus.popleft()
        tag = f"{task}/{arm_name}/s{seed}"
        print(f"  START GPU {gpu_id}: {tag}")
        job = launch_one(task, arm_name, ckpt_path, seed, gpu_id, args)
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
            tag = f"{job['task']}/{job['arm']}/s{job['seed']}"
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
                task, arm_name, ckpt_path, seed = queue.popleft()
                gpu_id = free_gpus.popleft()
                tag2 = f"{task}/{arm_name}/s{seed}"
                print(f"  START GPU {gpu_id}: {tag2}")
                new_job = launch_one(task, arm_name, ckpt_path, seed,
                                     gpu_id, args)
                still_active.append(new_job)

        active = still_active

    total_elapsed = time.time() - t_start
    print(f"\nAll done in {total_elapsed/60:.1f}min "
          f"({len(completed)} OK, {len(failed)} failed, "
          f"{len(skipped)} skipped)")

    # Load all results
    all_results = []
    for task, arm_name, ckpt_path, seed in jobs:
        out_json = os.path.join(
            args.output_dir, f"{task}_{arm_name}_seed{seed}.json")
        if os.path.isfile(out_json):
            with open(out_json) as f:
                all_results.append(json.load(f))
        else:
            all_results.append({"checkpoint": ckpt_path, "task": task,
                                "seed": seed, "arm_name": arm_name,
                                "error": "no output", "eval_results": {}})
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run fine-tune benchmark battery (generative)")
    parser.add_argument(
        "--checkpoints", nargs="+", required=True,
        help="arm=path pairs, e.g. baseline=/path/to/checkpoint-10000")
    parser.add_argument("--tasks", nargs="+",
                        default=list(TASK_CONFIGS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--output-dir", default="temp/finetune")
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--num-gpus", type=int, default=None)
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

    jobs = []
    for task in args.tasks:
        for arm_name, ckpt_path in arm_paths.items():
            for seed in args.seeds:
                jobs.append((task, arm_name, ckpt_path, seed))

    print(f"Total jobs: {len(jobs)} "
          f"({len(args.tasks)} tasks × {len(arm_paths)} arms × "
          f"{len(args.seeds)} seeds)")

    if args.num_gpus is None:
        gpu_ids = detect_gpus()
        args.num_gpus = len(gpu_ids)
        print(f"Auto-detected {args.num_gpus} GPUs")
    else:
        gpu_ids = list(range(args.num_gpus))

    t_start = time.time()
    all_results = run_parallel(jobs, args, gpu_ids)

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed/3600:.1f}h")
    build_summary([r for r in all_results if "error" not in r], args.output_dir)
    failed_results = [result for result in all_results if "error" in result]
    if failed_results:
        print(f"ERROR: {len(failed_results)} finetune jobs have no valid output")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
