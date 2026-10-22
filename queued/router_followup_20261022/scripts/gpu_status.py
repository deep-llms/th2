"""Read-only NVIDIA GPU inspection. This utility never signals any process."""
import argparse
import csv
import io
import json
import subprocess


def query(fields, kind):
    result = subprocess.run(
        ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, timeout=30,
    )
    if result.returncode:
        raise RuntimeError(f"nvidia-smi {kind} query failed (exit {result.returncode})")
    return [[cell.strip() for cell in row] for row in
            csv.reader(io.StringIO(result.stdout)) if row]


def snapshot(indices=None):
    gpu_rows = query("index,uuid,name,memory.total,memory.used,utilization.gpu", "gpu")
    if not gpu_rows:
        raise RuntimeError("nvidia-smi reported no GPUs")
    gpus = {}
    for row in gpu_rows:
        if len(row) != 6:
            raise RuntimeError("Malformed GPU query output")
        index, uuid, name, total, used, utilization = row
        index = int(index)
        if index in gpus:
            raise RuntimeError("Duplicate GPU index")
        gpus[index] = {"index": index, "uuid": uuid, "name": name,
                       "memory_total_mib": float(total), "memory_used_mib": float(used),
                       "utilization_percent": float(utilization), "pids": []}
    selected = sorted(gpus) if indices is None else list(indices)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("GPU selection must be nonempty and unique")
    if any(index not in gpus for index in selected):
        raise ValueError("Requested physical GPU index is not present")
    by_uuid = {gpu["uuid"]: gpu for gpu in gpus.values()}
    if len(by_uuid) != len(gpus):
        raise RuntimeError("Duplicate GPU UUID; device mapping is unverified")
    for row in query("gpu_uuid,pid", "compute-apps"):
        if len(row) != 2 or row[0] not in by_uuid:
            # Fail closed on unsupported MIG/UUID layouts; do not report free.
            raise RuntimeError("Unrecognized compute-process GPU UUID/layout")
        pid = int(row[1])
        if pid <= 0:
            raise RuntimeError("Invalid compute PID")
        by_uuid[row[0]]["pids"].append(pid)
    return [gpus[index] for index in selected]


def require_free(indices=None):
    status = snapshot(indices)
    occupied = [(gpu["index"], gpu["pids"]) for gpu in status if gpu["pids"]]
    if occupied:
        raise RuntimeError(f"GPU compute processes present; refusing to launch: {occupied}")
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, help="Physical GPU indices; default: all")
    parser.add_argument("--require-free", action="store_true")
    args = parser.parse_args()
    try:
        status = require_free(args.gpus) if args.require_free else snapshot(args.gpus)
    except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f"GPU status unverified/not free: {error}\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
