"""Run an explicitly configured sequence of jobs; never kill unknown GPU jobs.

Python 3.11+, POSIX. Commands are argv lists, not implicit shell strings.
Only a successful exit plus verified artifacts produces complete.json.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from scripts.gpu_status import require_free


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_name(path.name + ".part")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def output_path(run_dir, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or relative in ("", "."):
        raise ValueError("Artifact paths must be relative files beneath the run directory")
    resolved = (run_dir / path).resolve()
    if not resolved.is_relative_to(run_dir.resolve()):
        raise ValueError("Artifact path escapes run directory")
    return resolved


def load_jobs(path):
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict) or set(config) != {"jobs"}:
        raise ValueError("Config must contain only a jobs list")
    jobs = config["jobs"]
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Configure at least one job")
    names = set()
    reserved = {"complete.json", "run.json", "jobs.snapshot.json"}
    for job in jobs:
        if not isinstance(job, dict) or set(job) - {
            "name", "argv", "gpus", "timeout_seconds", "required_outputs"
        }:
            raise ValueError("Invalid job fields")
        name = job.get("name", "")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) or name in names:
            raise ValueError("Job names must be unique safe identifiers")
        names.add(name)
        argv = job.get("argv")
        if not isinstance(argv, list) or not argv or any(not isinstance(a, str) or not a for a in argv):
            raise ValueError(f"{name}: argv must be a nonempty string list")
        timeout = job.get("timeout_seconds")
        if timeout is not None and (type(timeout) not in (int, float) or
                                    not math.isfinite(timeout) or timeout <= 0):
            raise ValueError(f"{name}: timeout must be positive and finite")
        if "gpus" in job:
            gpus = job["gpus"]
            if not isinstance(gpus, list) or not gpus or any(type(i) is not int or i < 0 for i in gpus) or len(set(gpus)) != len(gpus):
                raise ValueError(f"{name}: gpus must be unique nonnegative physical indices")
        outputs = job.get("required_outputs")
        if not isinstance(outputs, list) or not outputs:
            raise ValueError(f"{name}: declare at least one required output")
        for artifact in outputs:
            if not isinstance(artifact, dict) or set(artifact) - {"path", "sha256", "json_equals"}:
                raise ValueError("Invalid artifact specification")
            relative = artifact.get("path")
            if not isinstance(relative, str):
                raise ValueError("Artifact requires a path")
            output_path(Path.cwd(), relative)
            if Path(relative).parts[0] in reserved or Path(relative).suffix == ".log":
                raise ValueError("Runner metadata/logs cannot count as job artifacts")
            if "sha256" in artifact and (not isinstance(artifact["sha256"], str) or
                    not re.fullmatch(r"[a-fA-F0-9]{64}", artifact["sha256"])):
                raise ValueError("Invalid SHA256")
            if "json_equals" in artifact and not isinstance(artifact["json_equals"], dict):
                raise ValueError("json_equals must be an object of expected top-level fields")
    return jobs


def require_fresh_outputs(job, run_dir):
    """A later job must not claim an earlier job's output, even in a fresh run."""
    for spec in job["required_outputs"]:
        path = output_path(run_dir, spec["path"])
        # Check the original spelling too: resolve() follows dangling symlinks.
        if os.path.lexists(run_dir / spec["path"]) or os.path.lexists(path):
            raise ValueError(f"Output already exists before {job['name']}: {spec['path']}")


def verify_outputs(job, run_dir):
    verified = []
    for spec in job["required_outputs"]:
        path = output_path(run_dir, spec["path"])
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing/empty artifact: {spec['path']}")
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if "sha256" in spec and digest != spec["sha256"].lower():
            raise ValueError(f"Artifact checksum mismatch: {spec['path']}")
        if "json_equals" in spec:
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict) or any(key not in data or data[key] != value
                    for key, value in spec["json_equals"].items()):
                raise ValueError(f"Artifact JSON contract failed: {spec['path']}")
        verified.append({"path": spec["path"], "sha256": digest, "bytes": path.stat().st_size})
    return verified


def stop_owned_process(proc):
    """Signal only the session/process group created for THIS child by Popen.

    Never derive a group from nvidia-smi PIDs. If ownership changed, refuse.
    Daemonized/detached workloads are not supported by this runner.
    """
    if proc.poll() is not None:
        return
    if proc.pid <= 1 or os.getpgid(proc.pid) != proc.pid or os.getsid(proc.pid) != proc.pid:
        raise RuntimeError("Refusing to stop child with unexpected session ownership")
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)


def run_jobs(jobs, project_dir, run_dir):
    project_dir, run_dir = project_dir.resolve(), run_dir.resolve()
    if not project_dir.is_dir():
        raise ValueError("Project directory is missing")
    # A fresh run root prevents a previous run's artifacts from faking success.
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "started_at": now(), "jobs": []}
    write_json(run_dir / "jobs.snapshot.json", {"jobs": jobs})
    write_json(run_dir / "run.json", report)
    try:
        for job in jobs:
            record = {"name": job["name"], "started_at": now(), "status": "running"}
            report["jobs"].append(record)
            write_json(run_dir / "run.json", report)
            require_fresh_outputs(job, run_dir)
            if "gpus" in job:
                require_free(job["gpus"])
            replacements = {"{python}": sys.executable, "{run_dir}": str(run_dir),
                            "{project_dir}": str(project_dir)}
            argv = []
            for arg in job["argv"]:
                for key, value in replacements.items():
                    arg = arg.replace(key, value)
                argv.append(arg)
            env = os.environ.copy()
            if "gpus" in job:
                env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, job["gpus"]))
            else:
                # CPU jobs do not accidentally acquire every visible GPU.
                env["CUDA_VISIBLE_DEVICES"] = ""
            print(f"START {job['name']}", flush=True)
            start = time.monotonic()
            with (run_dir / f"{job['name']}.log").open("x") as log:
                proc = subprocess.Popen(argv, cwd=project_dir, env=env,
                                        stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                try:
                    returncode = proc.wait(timeout=job.get("timeout_seconds"))
                except BaseException:
                    stop_owned_process(proc)
                    raise
            record["returncode"] = returncode
            record["elapsed_seconds"] = time.monotonic() - start
            if returncode != 0:
                raise RuntimeError(f"{job['name']} exited with code {returncode}")
            record["artifacts"] = verify_outputs(job, run_dir)
            if "gpus" in job:
                require_free(job["gpus"])
            record.update(status="ok", finished_at=now())
            print(f"OK {job['name']}", flush=True)
        report.update(status="ok", finished_at=now())
        write_json(run_dir / "run.json", report)
        write_json(run_dir / "complete.json", report)
        return 0
    except BaseException as error:
        if report["jobs"] and report["jobs"][-1]["status"] == "running":
            report["jobs"][-1]["status"] = "failed"
        report.update(status="failed", finished_at=now(), error=str(error))
        write_json(run_dir / "run.json", report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    try:
        jobs = load_jobs(args.config)
        if args.list:
            for job in jobs:
                print(job["name"])
            return 0
        if args.run_dir is None:
            parser.error("--run-dir is required unless --list is used")
        if os.name != "posix":
            raise ValueError("Job execution requires POSIX session/process-group support")
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Runner interrupted by signal {signum}")
        signal.signal(signal.SIGTERM, interrupted)
        return run_jobs(jobs, args.project_dir, args.run_dir)
    except KeyboardInterrupt:
        print("Interrupted; no completion marker", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
