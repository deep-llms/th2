"""Run experiments sequentially on multi-GPU machines.

Each experiment is a shell command string. The runner handles GPU cleanup
between runs, monitors a progress CSV (if specified), and terminates
each experiment when it reaches a target step.

Usage:
    python run_experiments.py --stop-at-step 6500
    python run_experiments.py --experiments 0 2    # run 1st and 3rd only
    python run_experiments.py --list

To add experiments: edit EXPERIMENT_COMMANDS below.
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import time

logger = logging.getLogger(__name__)

# =====================================================================
# EXPERIMENT COMMANDS — edit this list to define your experiments
#
# Each entry is a dict with:
#   "name": short label for logging and output files
#   "cmd": the full shell command to run
#   "output_dir": where checkpoints go (monitored for --stop-at-step)
# =====================================================================

OUT_BASE = "/opt/dlami/nvme/sparse_emb_outputs"

EXPERIMENT_COMMANDS = [
    {
        "name": "original_ant",
        "cmd": "bash scripts/train_original_ant.sh",
        "output_dir": f"{OUT_BASE}/original_ant",
    },
    {
        "name": "ant_ours",
        "cmd": "bash scripts/train_ant_ours.sh",
        "output_dir": f"{OUT_BASE}/ant_ours",
    },
    {
        "name": "v2_attn",
        "cmd": "bash scripts/train_v2_attn.sh",
        "output_dir": f"{OUT_BASE}/v2_attn",
    },
    {
        "name": "baseline",
        "cmd": "bash scripts/train_qwen3_0.6b_baseline.sh",
        "output_dir": f"{OUT_BASE}/baseline",
    },
]


# =====================================================================
# GPU management
# =====================================================================

def ensure_gpus_free(max_attempts=10, sleep_between=30):
    """Wait until all GPUs have 0 processes. Kill stragglers if needed."""
    for attempt in range(max_attempts):
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            logger.warning(f"    nvidia-smi failed (code {result.returncode})")
            time.sleep(sleep_between)
            continue

        pids = list(dict.fromkeys(
            p.strip() for p in result.stdout.strip().split("\n") if p.strip()
        ))

        if not pids:
            logger.info(f"    GPUs are free (attempt {attempt + 1})")
            return True

        logger.info(f"    GPUs in use ({len(pids)} processes), attempt {attempt + 1}/{max_attempts}")

        killed_groups = set()
        for pid in pids:
            try:
                pid_int = int(pid)
                pgid = os.getpgid(pid_int)
                if pgid not in killed_groups:
                    os.killpg(pgid, signal.SIGTERM)
                    killed_groups.add(pgid)
            except (ProcessLookupError, PermissionError, ValueError, OSError):
                pass

        time.sleep(5)

        for pid in pids:
            try:
                os.killpg(os.getpgid(int(pid)), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, ValueError, OSError):
                pass
            try:
                os.kill(int(pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, ValueError, OSError):
                pass

        time.sleep(max(0, sleep_between - 5))

    logger.error("    Failed to free GPUs after all attempts!")
    return False


# =====================================================================
# Step monitoring
# =====================================================================

def _latest_checkpoint_step(output_dir):
    """Return the highest step number from checkpoint-* dirs, or -1."""
    if not output_dir or not os.path.isdir(output_dir):
        return -1
    steps = []
    for d in os.listdir(output_dir):
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d)):
            try:
                steps.append(int(d.split("-")[1]))
            except (IndexError, ValueError):
                pass
    return max(steps) if steps else -1


def wait_for_step(proc, output_dir, target_step, poll_interval=30):
    """Monitor checkpoint directories for step progress. Kill when target reached.

    The training script saves checkpoints every --save_steps to output_dir/.
    This watches for checkpoint-{step}/ directories and terminates training
    once the step exceeds the target. The LR schedule / lambda ramp are still
    configured for the full run — only the process is stopped early.
    """
    while proc.poll() is None:
        time.sleep(poll_interval)
        last_step = _latest_checkpoint_step(output_dir)
        if last_step < 0:
            continue
        if last_step >= target_step:
            logger.info(f"    Reached step {last_step} >= {target_step}, waiting 120s for save...")
            time.sleep(120)
            logger.info(f"    Sending SIGTERM...")
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except OSError:
                pass
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except OSError:
                    pass
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
            return last_step
    return _latest_checkpoint_step(output_dir)


# =====================================================================
# Experiment runner
# =====================================================================

def run_experiment(exp, stop_at_step, log_dir):
    """Run a single experiment."""
    name = exp["name"]
    cmd = exp["cmd"]
    output_dir = exp.get("output_dir")

    log_path = os.path.join(log_dir, f"{name}.log")

    logger.info(f"  Ensuring GPUs are free...")
    if not ensure_gpus_free():
        logger.error(f"  SKIP {name}: could not free GPUs")
        return {"name": name, "status": "SKIPPED", "elapsed": 0}

    logger.info(f"  START: {name}")
    logger.info(f"  Command: {cmd}")
    if stop_at_step:
        logger.info(f"  Will stop at step {stop_at_step} (monitoring {output_dir})")
    start = time.time()

    env = os.environ.copy()
    env["NCCL_NVLS_ENABLE"] = "0"
    env["WANDB_MODE"] = env.get("WANDB_MODE", "offline")

    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            cmd, shell=True, stdout=log_file, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid, env=env,
        )

        if stop_at_step and output_dir:
            last_step = wait_for_step(proc, output_dir, stop_at_step)
        else:
            proc.wait()
            last_step = _latest_checkpoint_step(output_dir) if output_dir else -1

    elapsed = time.time() - start

    if stop_at_step and last_step >= stop_at_step:
        status = f"STOPPED at step {last_step}"
    elif proc.returncode == 0:
        status = "OK"
    elif proc.returncode == -signal.SIGTERM or proc.returncode == -signal.SIGKILL:
        status = f"KILLED at step {last_step}"
    else:
        status = f"FAILED (code {proc.returncode})"

    logger.info(f"  DONE: {name} — {status}  [{elapsed:.0f}s]")
    return {"name": name, "status": status, "elapsed": elapsed}


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Run experiments sequentially")
    parser.add_argument("--experiments", nargs="+", type=int, default=None,
                        help="Indices of experiments to run (default: all)")
    parser.add_argument("--stop-at-step", type=int, default=None,
                        help="Stop each experiment at this step (monitors checkpoint dirs in output_dir)")
    parser.add_argument("--log-dir", default=".",
                        help="Directory for per-experiment log files (default: .)")
    parser.add_argument("--list", action="store_true",
                        help="List available experiments and exit")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(os.path.join(args.log_dir, "experiments.log"), mode="w")],
    )

    if args.list:
        for i, exp in enumerate(EXPERIMENT_COMMANDS):
            print(f"  [{i}] {exp['name']}: {exp['cmd'][:80]}...")
        return

    if not EXPERIMENT_COMMANDS:
        print("No experiments configured. Edit EXPERIMENT_COMMANDS in this file.")
        return

    if args.experiments is not None:
        to_run = [EXPERIMENT_COMMANDS[i] for i in args.experiments]
    else:
        to_run = EXPERIMENT_COMMANDS

    start_time = time.time()

    logger.info(f"Running {len(to_run)} experiments sequentially")
    if args.stop_at_step:
        logger.info(f"Stop at step: {args.stop_at_step}")

    completed = []
    for i, exp in enumerate(to_run):
        logger.info(f"\n[{i+1}/{len(to_run)}] {exp['name']}")
        result = run_experiment(exp, args.stop_at_step, args.log_dir)
        completed.append(result)

        if i < len(to_run) - 1:
            logger.info(f"    Waiting 10s...")
            time.sleep(10)

    total = time.time() - start_time
    logger.info(f"\nAll {len(completed)} experiments done in {total:.0f}s")
    logger.info("=" * 50)
    for job in completed:
        logger.info(f"  {job['name']}: {job['status']} [{job['elapsed']:.0f}s]")


if __name__ == "__main__":
    main()
