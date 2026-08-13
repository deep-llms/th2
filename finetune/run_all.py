"""Run the full fine-tuning benchmark battery sequentially.

For each (task, mode, arm, seed): load model, train, evaluate, save JSON,
free GPU, sleep 30s, verify GPU is free, then continue.

Writes per-run JSONs to --output-dir, and a final merged summary table.

Usage:
  python finetune/run_all.py \
      --checkpoints baseline=/path/baseline/checkpoint-10000 \
                     lowrank=/path/lowrank/checkpoint-10000 \
                     ant_ours=/path/ant_ours/checkpoint-10000 \
      --tasks ag_news sst2 \
      --modes full probe \
      --seeds 42 123 456 \
      --output-dir temp/finetune \
      --device cuda
"""

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finetune.tasks import TASK_CONFIGS
from finetune.train import train_one


def check_gpu_free():
    """Return True if all GPUs have 0 MiB used."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True)
        return all(int(x.strip()) == 0 for x in r.stdout.strip().split("\n")
                   if x.strip())
    except Exception:
        return True


def wait_gpu_free(timeout=120):
    """Wait until GPUs are free, up to timeout seconds."""
    for _ in range(timeout // 5):
        if check_gpu_free():
            return True
        time.sleep(5)
    return False


def build_summary(results, output_dir):
    """Build a markdown summary table from all result JSONs."""
    by_task_mode = {}
    for r in results:
        key = (r["task"], r["mode"])
        arm = os.path.basename(os.path.dirname(
            os.path.normpath(r["checkpoint"])))
        by_task_mode.setdefault(key, {}).setdefault(arm, []).append(r)

    lines = ["# Fine-tune Benchmark Results\n"]

    all_arms = sorted({os.path.basename(os.path.dirname(
        os.path.normpath(r["checkpoint"]))) for r in results})

    for (task, mode), arms_data in sorted(by_task_mode.items()):
        lines.append(f"\n## {task} — {mode}\n")

        # Find all test splits from any arm
        test_keys = set()
        for arm, runs in arms_data.items():
            for r in runs:
                test_keys.update(r["test_results"].keys())
        test_keys = sorted(test_keys)

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

    total = (len(args.tasks) * len(args.modes) * len(arm_paths)
             * len(args.seeds))
    done = 0
    all_results = []
    t_start = time.time()

    for task in args.tasks:
        for mode in args.modes:
            for arm_name, ckpt_path in arm_paths.items():
                for seed in args.seeds:
                    done += 1
                    tag = f"{task}/{mode}/{arm_name}/seed{seed}"
                    print(f"\n{'='*60}")
                    print(f"  [{done}/{total}] {tag}")
                    print(f"{'='*60}")

                    out_json = os.path.join(
                        args.output_dir,
                        f"{task}_{mode}_{arm_name}_seed{seed}.json")
                    if os.path.isfile(out_json):
                        print(f"  SKIP (already exists): {out_json}")
                        with open(out_json) as f:
                            all_results.append(json.load(f))
                        continue

                    try:
                        result = train_one(
                            ckpt_path, task, mode, seed,
                            args.device, args.output_dir,
                            args.tokenizer_name, args.num_workers)
                        all_results.append(result)
                    except Exception as e:
                        print(f"  FAILED: {e}")
                        import traceback
                        traceback.print_exc()
                        all_results.append({
                            "checkpoint": ckpt_path, "task": task,
                            "mode": mode, "seed": seed, "error": str(e),
                            "test_results": {},
                        })

                    print(f"  GPU cleanup: sleeping 30s...")
                    time.sleep(30)
                    if not wait_gpu_free():
                        print(f"  WARNING: GPU not free after 120s")

    elapsed = time.time() - t_start
    print(f"\nAll {total} runs done in {elapsed/3600:.1f}h")

    build_summary(all_results, args.output_dir)


if __name__ == "__main__":
    main()
