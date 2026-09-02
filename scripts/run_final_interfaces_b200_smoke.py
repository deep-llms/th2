#!/usr/bin/env python3
"""Run all final-interface CUDA smoke tests and a 50-step RankLift gate.

This orchestrator is intentionally Python so subprocess ownership is explicit.
It verifies and stops only the current runner burn, runs the five-arm
production-shape DDP interface smoke, runs a real 8-GPU Qwen RankLift training
smoke, validates the results, and restores the communicating high-memory burn.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


EXPECTED_ARMS = {
    "ranklift_tied_c124_m460": 19_396_128,
    "funneling_tied_r128": 19_579_904,
    "define_tied_n112_k1724": 19_579_556,
    "slim_tied_k4_m76484": 19_579_904,
    "tt_tied_r219": 19_496_256,
}


def die(message: str) -> None:
    raise RuntimeError(message)


def output(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def gpu_pids() -> list[int]:
    value = output(
        "nvidia-smi", "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    )
    return sorted({int(line.strip()) for line in value.splitlines() if line.strip()})


def parent(pid: int) -> int:
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("PPid:"):
            return int(line.split()[1])
    die(f"missing PPid for {pid}")


def descendant(pid: int, ancestor: int) -> bool:
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == ancestor:
            return True
        seen.add(pid)
        pid = parent(pid)
    return False


def require_b200_node() -> None:
    names = output(
        "nvidia-smi", "--query-gpu=name", "--format=csv,noheader"
    ).splitlines()
    if len(names) != 8 or any("B200" not in name for name in names):
        die(f"expected eight B200 GPUs, found {names}")


def require_free(stage: str) -> None:
    subprocess.run(["nvidia-smi"], check=True)
    pids = gpu_pids()
    if pids:
        die(f"GPU processes remain {stage}: {pids}")
    print(f"ALL_EIGHT_B200_GPUS_FREE {stage}", flush=True)


class BurnManager:
    def __init__(self, source: Path) -> None:
        self.source = source
        self.target = Path("/tmp/llm_pretrain_burn.py")
        self.log = Path("/tmp/llm_pretrain_burn_all_gpus.log")
        self.pid_file = Path("/tmp/llm_pretrain_burn_launcher.pid")
        self.python = Path("/usr/bin/python3")
        self.restored = False

    def stop_verified_current(self) -> None:
        pids = gpu_pids()
        if len(pids) != 8:
            die(f"expected eight current burn workers, found {pids}")
        if not self.pid_file.is_file():
            die(f"missing burn launcher file: {self.pid_file}")
        launcher = int(self.pid_file.read_text().strip())
        if launcher == 1 or not Path(f"/proc/{launcher}").exists():
            die(f"invalid burn launcher PID: {launcher}")
        args = Path(f"/proc/{launcher}/cmdline").read_bytes().replace(b"\0", b" ")
        if str(self.target).encode() not in args:
            die(f"unexpected burn launcher command: {args!r}")
        if not all(pid != 1 and descendant(pid, launcher) for pid in pids):
            die(f"GPU PIDs are not all burn descendants: launcher={launcher}, pids={pids}")
        app_rows = [
            [field.strip() for field in row]
            for row in csv.reader(output(
                "nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ).splitlines()) if row
        ]
        if len(app_rows) != 8 or len({uuid for uuid, _ in app_rows}) != 8:
            die(f"burn does not own exactly one process per GPU: {app_rows}")
        for pid in pids:
            os.kill(pid, signal.SIGKILL)
        try:
            os.kill(launcher, signal.SIGKILL)
        except ProcessLookupError:
            pass
        time.sleep(30)
        require_free("AFTER_VERIFIED_BURN_STOP")

    def start_and_verify(self) -> None:
        require_free("BEFORE_BURN_RESTORE")
        shutil.copyfile(self.source, self.target)
        os.chmod(self.target, 0o644)
        if self.source.read_bytes() != self.target.read_bytes():
            die("installed burn differs from repository source")
        self.log.unlink(missing_ok=True)
        self.pid_file.unlink(missing_ok=True)
        handle = self.log.open("wb")
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29500",
            "NCCL_DEBUG": "WARN",
        })
        process = subprocess.Popen(
            [str(self.python), "-u", str(self.target)],
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        handle.close()
        self.pid_file.write_text(f"{process.pid}\n")
        for _ in range(240):
            if process.poll() is not None:
                die(f"burn exited during startup:\n{self.log.read_text()}")
            text = self.log.read_text(errors="replace") if self.log.exists() else ""
            if text.count("gpu_burn_ready") == 8 and len(gpu_pids()) == 8:
                break
            time.sleep(1)
        else:
            die(f"burn did not become ready:\n{self.log.read_text()}")
        for _ in range(120):
            text = self.log.read_text(errors="replace")
            if "gpu_burn_progress" in text:
                break
            time.sleep(1)
        else:
            die("burn produced no synchronized progress marker")
        text = self.log.read_text(errors="replace")
        if text.count("world_size=8") != 8:
            die("not every burn rank joined the eight-rank NCCL group")
        if len(gpu_pids()) != 8:
            die("burn does not own eight GPU processes after startup")
        self.restored = True
        subprocess.run([
            "nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], check=True)
        print("CORRECT_EIGHT_GPU_COMMUNICATING_BURN_RESTORED", flush=True)


def run_logged(command: list[str], log: Path, *, env: dict[str, str]) -> float:
    started = time.monotonic()
    with log.open("w") as handle:
        subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
            env=env,
        )
    elapsed = time.monotonic() - started
    print(f"COMMAND_OK seconds={elapsed:.1f} log={log}", flush=True)
    return elapsed


def validate_interface_report(path: Path) -> None:
    report = json.loads(path.read_text())
    if report.get("status") != "PASS" or report.get("world_size") != 8:
        die(f"bad interface smoke report: {report}")
    rows = {row["arm"]: row for row in report["arms"]}
    if set(rows) != set(EXPECTED_ARMS):
        die(f"wrong interface arms: {rows.keys()}")
    for name, parameters in EXPECTED_ARMS.items():
        row = rows[name]
        if row.get("status") != "PASS" or row.get("parameters") != parameters:
            die(f"bad interface result for {name}: {row}")
        if row.get("world_size") != 8 or row.get("nccl_rank_sum") != 36:
            die(f"bad DDP/NCCL result for {name}: {row}")
        if not all(math.isfinite(float(loss)) for loss in row["losses"]):
            die(f"non-finite smoke loss for {name}: {row['losses']}")
    print("FIVE_PRODUCTION_INTERFACE_SMOKES_PASS", flush=True)


def validate_training_smoke(output_dir: Path) -> dict:
    results = json.loads((output_dir / "train_results.json").read_text())
    state = json.loads((output_dir / "checkpoint-50" / "trainer_state.json").read_text())
    if int(state["global_step"]) != 50:
        die(f"RankLift smoke stopped at {state['global_step']}, expected 50")
    losses = [float(row["loss"]) for row in state["log_history"] if "loss" in row]
    if not losses or not all(math.isfinite(value) for value in losses):
        die(f"bad RankLift smoke losses: {losses}")
    steps_per_second = float(results["train_steps_per_second"])
    # The completed matched GroupReduce run was about 0.3 steps/s.  Do not
    # authorize a full RankLift run if a warmed 50-step gate is materially
    # slower than that already-unsatisfactory control.
    if steps_per_second < 0.30:
        die(
            f"RankLift throughput gate failed: {steps_per_second:.4f} steps/s < 0.30"
        )
    if losses[-1] > 12.5:
        die(f"RankLift loss gate failed: final logged loss {losses[-1]:.4f}")
    summary = {
        "status": "PASS",
        "global_step": 50,
        "train_runtime": float(results["train_runtime"]),
        "train_steps_per_second": steps_per_second,
        "first_logged_loss": losses[0],
        "last_logged_loss": losses[-1],
    }
    (output_dir.parent / "ranklift_training_smoke_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    print("RANKLIFT_REAL_QWEN_50_STEP_GATE_PASS", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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
    ranklift_smoke = smoke_root / "ranklift_qwen_50step"
    interface_report = smoke_root / "interface_report.json"
    interface_log = smoke_root / "interface_smoke.log"
    training_log = smoke_root / "ranklift_qwen_50step.log"
    accelerate_source = project / "resources/accelerate_config.yaml"
    accelerate_target = Path(
        "/mnt/local/.cache/huggingface/accelerate/default_config.yaml"
    )

    os.chdir(project)
    require_b200_node()
    for path in (
        python, model_dir / "config.json", model_dir / "tokenizer.json",
        accelerate_source, project / "resources/llm_pretrain_burn.py",
    ):
        if not path.is_file() or path.stat().st_size == 0:
            die(f"missing required file: {path}")
    for language in ("en", "ar", "de", "ru", "vi", "zh"):
        if not (data_dir / language).is_dir():
            die(f"missing training language directory: {data_dir / language}")
    if smoke_root.exists():
        die(f"refusing to reuse smoke output: {smoke_root}")
    smoke_root.mkdir(parents=True)

    burn = BurnManager(project / "resources/llm_pretrain_burn.py")
    try:
        burn.stop_verified_current()
        accelerate_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(accelerate_source, accelerate_target)
        if accelerate_source.read_bytes() != accelerate_target.read_bytes():
            die("Accelerate configuration copy mismatch")
        text = accelerate_target.read_text()
        for line in (
            "distributed_type: MULTI_GPU",
            "mixed_precision: bf16",
            "num_processes: 8",
        ):
            if line not in text.splitlines():
                die(f"Accelerate configuration missing: {line}")
        time.sleep(30)
        require_free("BEFORE_INTERFACE_SMOKE")

        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "WANDB_MODE": "offline",
            "NCCL_NVLS_ENABLE": "0",
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        })
        run_logged([
            str(python), "-u", "scripts/smoke_final_compressed_interfaces_gpu.py",
            "--output", str(interface_report),
        ], interface_log, env=env)
        validate_interface_report(interface_report)
        time.sleep(30)
        require_free("AFTER_INTERFACE_SMOKE")

        command = [
            str(python), "-m", "accelerate.commands.launch",
            "train_compositional.py",
            "--config_name", str(model_dir),
            "--tokenizer_name", str(model_dir),
            "--data_dir", str(data_dir),
            "--block_size", "2048",
            "--preprocessing_num_workers", "160",
            "--seed", "42",
            "--bf16",
            "--ddp_timeout", "21600",
            "--ddp_find_unused_parameters", "false",
            "--per_device_train_batch_size", "16",
            "--gradient_accumulation_steps", "4",
            "--max_steps", "50",
            "--learning_rate", "3e-4",
            "--lr_scheduler_type", "cosine_with_min_lr",
            "--lr_scheduler_kwargs", '{"min_lr_rate": 0.1}',
            "--warmup_steps", "500",
            "--weight_decay", "0.1",
            "--adam_beta1", "0.9",
            "--adam_beta2", "0.95",
            "--max_grad_norm", "1.0",
            "--logging_steps", "10",
            "--save_steps", "50",
            "--dataloader_num_workers", "8",
            "--report_to", "none",
            "--output_dir", str(ranklift_smoke),
            "--run_name", "ranklift-b200-50step-smoke",
            "--arm", "ranklift",
            "--ranklift_code_dim", "124",
            "--ranklift_lift_dim", "336",
            "--ranklift_rms_eps", "1e-6",
            "--tie_output",
        ]
        run_logged(command, training_log, env=env)
        validate_training_smoke(ranklift_smoke)
        time.sleep(30)
        require_free("AFTER_RANKLIFT_TRAINING_SMOKE")
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

    print("FINAL_INTERFACES_AND_RANKLIFT_B200_SMOKE_COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"SMOKE_FAILED: {error}", file=sys.stderr, flush=True)
        raise
