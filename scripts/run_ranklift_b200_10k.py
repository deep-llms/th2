#!/usr/bin/env python3
"""Launch the validated RankLift experiment to checkpoint 10,000 on th2."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from run_final_interfaces_b200_smoke import (
    BurnManager,
    gpu_pids,
    require_b200_node,
    require_free,
    validate_interface_report,
)


REQUIRED_CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "trainer_state.json",
    "optimizer.pt",
    "scheduler.pt",
    "embedding.pt",
    "rng_state_0.pth",
    "rng_state_1.pth",
    "rng_state_2.pth",
    "rng_state_3.pth",
    "rng_state_4.pth",
    "rng_state_5.pth",
    "rng_state_6.pth",
    "rng_state_7.pth",
)


def die(message: str) -> None:
    raise RuntimeError(message)


def validate_ranklift_gate(smoke_root: Path) -> dict:
    validate_interface_report(smoke_root / "interface_report.json")
    summary = json.loads(
        (smoke_root / "ranklift_training_smoke_summary.json").read_text()
    )
    required = {
        "status": "PASS",
        "global_step": 50,
    }
    for key, expected in required.items():
        if summary.get(key) != expected:
            die(f"RankLift smoke summary has {key}={summary.get(key)!r}")
    speed = float(summary["train_steps_per_second"])
    losses = (
        float(summary["first_logged_loss"]),
        float(summary["last_logged_loss"]),
    )
    if speed < 0.30 or not all(math.isfinite(value) for value in losses):
        die(f"RankLift smoke gate is not acceptable: {summary}")
    if losses[-1] > 12.5:
        die(f"RankLift smoke loss gate is not acceptable: {summary}")
    print(f"RANKLIFT_50_STEP_GATE_REVALIDATED {json.dumps(summary, sort_keys=True)}")
    return summary


def validate_checkpoint(output_dir: Path, training_log: Path) -> None:
    checkpoint = output_dir / "checkpoint-10000"
    missing = [
        name for name in REQUIRED_CHECKPOINT_FILES
        if not (checkpoint / name).is_file()
        or (checkpoint / name).stat().st_size == 0
    ]
    if missing:
        die(f"checkpoint-10000 is incomplete: {missing}")
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    if int(state["global_step"]) != 10_000:
        die(f"wrong final step: {state['global_step']}")
    losses = [
        float(row["loss"]) for row in state.get("log_history", [])
        if "loss" in row
    ]
    if not losses or not all(math.isfinite(value) for value in losses):
        die(f"missing or non-finite RankLift training losses: {losses[-10:]}")
    log_text = training_log.read_text(errors="replace")
    fatal_markers = (
        "Traceback (most recent call last)",
        "CUDA out of memory",
        "OutOfMemoryError",
        "ChildFailedError",
        "ProcessExitedException",
        "Segmentation fault",
        "Bus error",
    )
    present = [marker for marker in fatal_markers if marker in log_text]
    if present:
        die(f"fatal marker(s) in RankLift training log: {present}")
    print(
        "RANKLIFT_CHECKPOINT_10000_VALID "
        f"logged_losses={len(losses)} first={losses[0]:.6f} last={losses[-1]:.6f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--output-base", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--python", required=True)
    args = parser.parse_args()

    project = Path(args.project_dir).resolve()
    output_base = Path(args.output_base).resolve()
    model_dir = Path(args.model_dir).resolve()
    data_dir = Path(args.data_dir).resolve()
    python = Path(args.python).resolve()
    smoke_root = output_base / "final_interfaces_smoke_20260902"
    output_dir = output_base / "ranklift_tied_c124_m460"
    log_dir = output_base / "logs" / "ranklift_tied_c124_m460_10k_20260902"
    training_log = log_dir / "ranklift_tied_c124_m460.log"
    accelerate_source = project / "resources/accelerate_config.yaml"
    accelerate_target = Path(
        "/mnt/local/.cache/huggingface/accelerate/default_config.yaml"
    )

    os.chdir(project)
    require_b200_node()
    for path in (
        python,
        model_dir / "config.json",
        model_dir / "tokenizer.json",
        accelerate_source,
        project / "resources/llm_pretrain_burn.py",
    ):
        if not path.is_file() or path.stat().st_size == 0:
            die(f"missing required file: {path}")
    for language in ("en", "ar", "de", "ru", "vi", "zh"):
        if not (data_dir / language).is_dir():
            die(f"missing training language directory: {data_dir / language}")
    validate_ranklift_gate(smoke_root)
    if output_dir.exists():
        die(f"refusing to reuse RankLift output directory: {output_dir}")
    if log_dir.exists():
        die(f"refusing to reuse RankLift log directory: {log_dir}")

    burn = BurnManager(project / "resources/llm_pretrain_burn.py")
    try:
        burn.stop_verified_current()
        accelerate_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(accelerate_source, accelerate_target)
        if accelerate_source.read_bytes() != accelerate_target.read_bytes():
            die("Accelerate configuration copy mismatch")
        config_lines = accelerate_target.read_text().splitlines()
        for line in (
            "distributed_type: MULTI_GPU",
            "mixed_precision: bf16",
            "num_processes: 8",
        ):
            if line not in config_lines:
                die(f"Accelerate configuration missing: {line}")
        time.sleep(30)
        require_free("BEFORE_RANKLIFT_10000")

        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "WANDB_MODE": "offline",
            "NCCL_NVLS_ENABLE": "0",
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "SPARSE_EMB_OUTPUT_BASE": str(output_base),
            "SPARSE_EMB_MODEL_DIR": str(model_dir),
            "SPARSE_EMB_DATA_DIR": str(data_dir),
            "SPARSE_EMB_PYTHON": str(python),
        })
        # Import only after setting SPARSE_EMB_OUTPUT_BASE: the runner resolves
        # experiment output paths at module import time.
        os.environ.update({
            key: env[key] for key in (
                "SPARSE_EMB_OUTPUT_BASE",
                "SPARSE_EMB_MODEL_DIR",
                "SPARSE_EMB_DATA_DIR",
                "SPARSE_EMB_PYTHON",
            )
        })
        if str(project) not in sys.path:
            sys.path.insert(0, str(project))
        import run_experiments

        matches = [
            index for index, experiment in enumerate(run_experiments.EXPERIMENT_COMMANDS)
            if experiment["name"] == "ranklift_tied_c124_m460"
        ]
        if len(matches) != 1:
            die(f"expected one RankLift registry entry, found {matches}")
        experiment = run_experiments.EXPERIMENT_COMMANDS[matches[0]]
        if Path(experiment["output_dir"]).resolve() != output_dir:
            die(f"RankLift registry output mismatch: {experiment['output_dir']}")

        log_dir.mkdir(parents=True)
        command = [
            str(python), "-u", "run_experiments.py",
            "--experiments", str(matches[0]),
            "--stop-at-step", "10000",
            "--log-dir", str(log_dir),
        ]
        print(f"RANKLIFT_10000_COMMAND {command}", flush=True)
        subprocess.run(command, check=True, env=env)
        validate_checkpoint(output_dir, training_log)
        time.sleep(30)
        require_free("AFTER_RANKLIFT_10000")
    finally:
        if not burn.restored:
            remaining = gpu_pids()
            if remaining:
                print(
                    "BURN_NOT_RESTORED_GPU_PROCESSES_REMAIN "
                    f"pids={remaining}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                burn.start_and_verify()

    print("RANKLIFT_TIED_C124_M460_CHECKPOINT_10000_COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"RANKLIFT_10000_FAILED: {error}", file=sys.stderr, flush=True)
        raise
