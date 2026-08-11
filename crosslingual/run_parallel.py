"""Run the cross-lingual battery on multiple checkpoints in parallel.

Same queue/worker pattern as eval/eval_parallel.py: one subprocess per
checkpoint pinned to a free GPU via CUDA_VISIBLE_DEVICES; when any GPU
finishes it immediately picks up the next queued checkpoint.

Each checkpoint gets its own output dir <output-base>/<label>/ (label =
"<model>@<step>", e.g. ant_ours@10000) and its own log file. After all jobs
finish, merge with crosslingual/merge_report.py.

Usage:
  python crosslingual/run_parallel.py \
      --checkpoints /path/ant_ours/checkpoint-10000 /path/baseline/checkpoint-10000 \
      --tests t6 t8 probe_b --mexa-sentences 500 \
      --output-base /opt/dlami/nvme/sparse_emb_outputs/crosslingual \
      --num-gpus 8
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from collections import deque

logger = logging.getLogger(__name__)


def label_for(ckpt):
    parent = os.path.basename(os.path.dirname(os.path.normpath(ckpt)))
    step = os.path.basename(os.path.normpath(ckpt)).replace("checkpoint-", "")
    return f"{parent}@{step}"


def launch(script, ckpt, gpu_id, args):
    label = label_for(ckpt)
    out_dir = os.path.join(args.output_base, label)
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, script,
           "--checkpoints", ckpt,
           "--label", label,
           "--tests", *args.tests,
           "--tokenizer-name", args.tokenizer_name,
           "--mexa-sentences", str(args.mexa_sentences),
           "--csls-k", str(args.csls_k),
           "--output-dir", out_dir]
    if args.eval_dir:
        cmd += ["--eval-dir", args.eval_dir]
    if args.flores_dir:
        cmd += ["--flores-dir", args.flores_dir]
    if args.translations:
        cmd += ["--translations", args.translations]

    log_path = os.path.join(args.output_base, f"{label}.log")
    log_file = open(log_path, "w")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                         env=env)
    return {"process": p, "ckpt": ckpt, "label": label, "gpu_id": gpu_id,
            "log_file": log_file, "log_path": log_path}


def main():
    parser = argparse.ArgumentParser(
        description="Parallel cross-lingual battery across GPUs")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--tests", nargs="+", default=["t6", "t8", "probe_b"])
    parser.add_argument("--tokenizer-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--translations", default=None)
    parser.add_argument("--eval-dir", default=None)
    parser.add_argument("--flores-dir", default=None)
    parser.add_argument("--csls-k", type=int, default=10)
    parser.add_argument("--mexa-sentences", type=int, default=500)
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="Default: auto-detect via nvidia-smi")
    parser.add_argument("--output-base", required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s",
                        datefmt="%H:%M:%S")

    if args.num_gpus is None:
        # nvidia-smi, not torch: keeps the parent free of CUDA init
        # (lesson from run_probe_tests.py)
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True, text=True)
            args.num_gpus = len([x for x in r.stdout.strip().split("\n")
                                 if x.strip()])
        except Exception:
            args.num_gpus = 1
        logger.info(f"Auto-detected {args.num_gpus} GPUs")

    checkpoints = [c for c in args.checkpoints if os.path.isdir(c)]
    for c in set(args.checkpoints) - set(checkpoints):
        logger.info(f"SKIPPING: {c} (not found)")
    if not checkpoints:
        logger.info("No valid checkpoints.")
        return

    logger.info(f"Running {args.tests} on {len(checkpoints)} checkpoints "
                f"across {args.num_gpus} GPUs")
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "run_crosslingual.py")
    os.makedirs(args.output_base, exist_ok=True)

    queue = deque(checkpoints)
    free_gpus = deque(range(args.num_gpus))
    active, completed = [], []
    start = time.time()

    while queue and free_gpus:
        job = launch(script, queue.popleft(), free_gpus.popleft(), args)
        logger.info(f"  START GPU {job['gpu_id']}: {job['label']}")
        active.append(job)

    while active:
        time.sleep(2)
        still = []
        for job in active:
            ret = job["process"].poll()
            if ret is None:
                still.append(job)
                continue
            job["log_file"].close()
            expected = os.path.join(args.output_base, job["label"],
                                    f"{job['label']}.json")
            if ret == 0 and not os.path.isfile(expected):
                status = "FAILED (exit 0 but no output JSON)"
            else:
                status = "OK" if ret == 0 else f"FAILED (code {ret})"
            logger.info(f"  DONE  GPU {job['gpu_id']}: {status} - "
                        f"{job['label']}  [{time.time() - start:.0f}s]")
            if "FAILED" in status:
                logger.info(f"        see {job['log_path']}")
            completed.append(job)
            free_gpus.append(job["gpu_id"])
            if queue:
                time.sleep(5)
                new = launch(script, queue.popleft(), free_gpus.popleft(),
                             args)
                logger.info(f"  START GPU {new['gpu_id']}: {new['label']}")
                still.append(new)
        active = still

    n_fail = sum(1 for j in completed if j["process"].returncode != 0)
    logger.info(f"All {len(completed)} done in {time.time() - start:.0f}s "
                f"({n_fail} failed)")


if __name__ == "__main__":
    main()
