"""Stage a frozen offline source/probe, verify Accelerate, then hand GPUs to the supervisor."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys

from scripts.gpu_process import inspect
from scripts.train_then_burn import BURN_HASH, GUARD_HASH, digest


def jobs_for(model, probe, config, port):
    return {"jobs": [
        {"name": "gpu-correctness", "gpus": [0], "timeout_seconds": 900,
         "argv": ["{python}", "-m", "efficient_llm.gpu_check", "--model", str(model),
                  "--probe", str(probe), "--output", "{run_dir}/gpu-check.json"],
         "required_outputs": [{"path": "gpu-check.json", "json_equals": {"status": "passed"}}]},
        {"name": "joint-groups", "gpus": list(range(8)), "timeout_seconds": 3600,
         "argv": ["{python}", "-m", "accelerate.commands.launch", "--config_file", str(config),
                  "--main_process_port", str(port), "--module", "efficient_llm.diagnostic", "run",
                  "--model", str(model), "--probe", str(probe), "--keep", "6", "4",
                  "--output", "{run_dir}/diagnostic.json"],
         "required_outputs": [{"path": "diagnostic.json", "json_equals": {"status": "passed", "world_size": 8}}]},
    ]}


def router_jobs_for(model, probe, config, port):
    launch = ["{python}", "-m", "accelerate.commands.launch", "--config_file", str(config),
              "--main_process_port", str(port), "--module"]
    return {"jobs": [
        {"name": "router-gpu-correctness", "gpus": [0], "timeout_seconds": 900,
         "argv": ["{python}", "-m", "efficient_llm.router_check", "--model", str(model),
                  "--probe", str(probe), "--output", "{run_dir}/gpu-check.json"],
         "required_outputs": [{"path": "gpu-check.json", "json_equals": {"status": "passed"}}]},
        {"name": "matched-router-pilot", "gpus": list(range(8)), "timeout_seconds": 3600,
         "argv": launch + ["efficient_llm.router_pilot", "--model", str(model), "--probe", str(probe),
                  "--keep", "6", "--stride", "16", "--steps", "512", "--feature-batch", "256",
                  "--output", "{run_dir}/pilot"],
         "required_outputs": [
             {"path": "pilot/training.json", "json_equals": {"status": "passed", "world_size": 8, "steps": 512}},
             {"path": "pilot/evaluation.json", "json_equals": {"status": "passed", "world_size": 8}}]},
        {"name": "cached-decode-profile", "gpus": list(range(8)), "timeout_seconds": 1200,
         "argv": launch + ["efficient_llm.decode_profile", "--model", str(model), "--steps", "32",
                  "--repeats", "3", "--output", "{run_dir}/decode-profile.json"],
         "required_outputs": [{"path": "decode-profile.json", "json_equals": {"status": "passed", "world_size": 8}}]},
    ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--predecessor-burn-script", type=Path, required=True)
    parser.add_argument("--study", choices=["diagnostic", "router"], default="diagnostic")
    parser.add_argument("--exclude-manifest", type=Path)
    args = parser.parse_args()
    source = Path.cwd().resolve()
    if socket.gethostname() != args.host or args.output.exists():
        raise ValueError("Wrong node or existing output")
    if args.study == "router" and (args.exclude_manifest is None or not args.exclude_manifest.is_file()):
        raise ValueError("Router pilot requires the previous held-out probe manifest")
    if args.output.parent.resolve() != Path("/mnt/local/_outputs/deep-llms_th2"):
        raise ValueError("Require the authorized external output root")
    if digest(args.predecessor_burn_script) != BURN_HASH:
        raise ValueError("Unknown predecessor burn")
    guard = Path("/mnt/local/_gpu_guard")
    if digest(guard/"gpu_guard.sh") != GUARD_HASH or (guard/"DISABLED").exists():
        raise ValueError("GPU guard changed or another owner holds it")
    for session in (args.session, args.session+"_burn"):
        if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode == 0:
            raise ValueError("Session already exists: " + session)
    manifest = json.loads((source/"resources/joint_source_manifest.json").read_text())
    for relative, expected in manifest.items():
        path = (source/relative).resolve()
        if not path.is_relative_to(source) or digest(path) != expected:
            raise ValueError("Source checksum mismatch: " + relative)
    args.output.mkdir()
    frozen = args.output/"source"
    for relative, expected in manifest.items():
        dest = frozen/relative; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/relative, dest)
        if digest(dest) != expected:
            raise ValueError("Snapshot mismatch: " + relative)
    (args.output/"source-manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1", WANDB_MODE="offline", OMP_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1", NCCL_NVLS_ENABLE="0",
        CUDA_VISIBLE_DEVICES="")
    # CPU preparation and authoritative weight/shard checks happen before any
    # GPU-worker stop. A preparation failure leaves the current burns alone.
    prepare = [sys.executable, "-m", "efficient_llm.diagnostic", "prepare", "--model", str(args.model),
        "--data-root", str(args.data_root), "--output", str(args.output/"probe"),
        "--contract", str(frozen/"resources/cx_english_20261020.json")]
    if args.study == "router":
        prepare += ["--calibration", "1024", "--evaluation", "256"]
    if args.exclude_manifest is not None:
        prepare += ["--exclude-manifest", str(args.exclude_manifest)]
    subprocess.run(prepare, cwd=frozen, env=env, check=True)
    from accelerate.commands.config.config_args import default_config_file
    config = frozen/"resources/accelerate_b200.yaml"
    cached = Path(default_config_file)
    cached.parent.mkdir(parents=True, exist_ok=True)
    if cached.exists():
        shutil.copyfile(cached, args.output/"previous-accelerate.yaml")
    shutil.copyfile(config, cached)
    if cached.read_bytes() != config.read_bytes():
        raise ValueError("Accelerate config copy differs")
    from accelerate.commands.config.config_args import load_config_from_file
    actual = load_config_from_file(str(cached))
    if actual.num_processes != 8 or actual.mixed_precision != "bf16" or actual.use_cpu:
        raise ValueError("Accelerate config differs from eight-GPU BF16 contract")
    result = subprocess.run([sys.executable, "-m", "accelerate.commands.env"], capture_output=True, text=True,
        cwd=frozen, env={**env, "CUDA_VISIBLE_DEVICES":"0,1,2,3,4,5,6,7"}, check=True)
    (args.output/"accelerate-env.log").write_text(result.stdout+result.stderr)
    print(result.stdout, flush=True)
    inspection = inspect()
    if len(inspection["gpus"]) != 8 or not all("B200" in g["name"] for g in inspection["gpus"]):
        raise ValueError("Expected eight B200 GPUs")
    inspection_path = args.output/"inspection.json"
    inspection_path.write_text(json.dumps(inspection, indent=2)+"\n")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]
    jobs = args.output/"jobs.json"
    factory = router_jobs_for if args.study == "router" else jobs_for
    jobs.write_text(json.dumps(factory(args.model, args.output/"probe", config, port), indent=2)+"\n")
    # Validate the manifest before creating the independent supervisor session.
    from run_experiments import load_jobs
    load_jobs(jobs)
    argv = ["env", *[f"{k}={v}" for k,v in env.items() if k in {
        "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY",
        "WANDB_MODE", "OMP_NUM_THREADS", "TOKENIZERS_PARALLELISM", "PYTHONDONTWRITEBYTECODE", "NCCL_NVLS_ENABLE", "CUDA_VISIBLE_DEVICES"}],
        sys.executable, "-u", "-m", "scripts.train_then_burn", "--config", str(jobs),
        "--output", str(args.output/"supervised"), "--inspection", str(inspection_path),
        "--host", args.host, "--burn-session", args.session+"_burn",
        "--predecessor-burn-script", str(args.predecessor_burn_script)]
    command = "exec " + shlex.join(argv) + " >" + shlex.quote(str(args.output/"supervisor.log")) + " 2>&1"
    subprocess.run(["tmux", "new-session", "-d", "-s", args.session, "-c", str(frozen), command], check=True)
    subprocess.run(["tmux", "set-option", "-w", "-t", args.session, "remain-on-exit", "on"], check=True)
    print("DIAGNOSTIC_SUBMITTED " + json.dumps({"study":args.study,"session":args.session,"output":str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
