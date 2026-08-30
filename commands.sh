#1 +30+a
#th2-audit-all-eight-supervised-gpu-burns-20260830-a01
set -euo pipefail

TASK_PYTHON=/mnt/local/conda-py311/envs/eval/bin/python3.11
TASK_RUNTIME=/tmp/run_phase1_four_model_eval_finetune_burn_2949bab98ea9.sh
TASK_PROJECT_DIR=/mnt/local/@PROJECT@

echo '=== read-only audit of all physical GPUs and compute processes ==='
"$TASK_PYTHON" - "$TASK_RUNTIME" "$TASK_PROJECT_DIR" <<'PY'
import csv
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

runtime = sys.argv[1]
project_dir = sys.argv[2]
expected_indices = set(range(8))


def run(*args):
    return subprocess.check_output(args, text=True).strip()


def rows(output):
    if not output:
        return []
    return [[field.strip() for field in row] for row in csv.reader(output.splitlines())]


def cmdline(pid):
    raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    return [part.decode(errors="replace") for part in raw.split(b"\0") if part]


def parent_pid(pid):
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("PPid:"):
            return int(line.split()[1])
    raise RuntimeError(f"missing PPid for PID {pid}")


gpu_rows = rows(run(
    "nvidia-smi",
    "--query-gpu=index,uuid,name,memory.used,utilization.gpu,power.draw",
    "--format=csv,noheader,nounits",
))
assert len(gpu_rows) == 8, f"expected 8 GPUs, found {len(gpu_rows)}: {gpu_rows}"

gpus = {}
for row in gpu_rows:
    assert len(row) == 6, f"unexpected GPU row: {row}"
    index = int(row[0])
    assert index not in gpus, f"duplicate GPU index {index}"
    assert "B200" in row[2], f"GPU {index} is not B200: {row[2]}"
    gpus[index] = {
        "uuid": row[1],
        "name": row[2],
        "memory": float(row[3]),
        "utilization": float(row[4]),
        "power": float(row[5]),
    }
assert set(gpus) == expected_indices, f"unexpected physical GPU indices: {sorted(gpus)}"
assert len({gpu["uuid"] for gpu in gpus.values()}) == 8, "GPU UUIDs are not unique"

uuid_to_index = {gpu["uuid"]: index for index, gpu in gpus.items()}
app_rows = rows(run(
    "nvidia-smi",
    "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
    "--format=csv,noheader,nounits",
))
assert len(app_rows) == 8, f"expected exactly 8 GPU compute applications, found {len(app_rows)}: {app_rows}"

apps = {}
all_pids = set()
for row in app_rows:
    assert len(row) == 4, f"unexpected compute-app row: {row}"
    gpu_uuid, pid_text, process_name, used_memory_text = row
    assert gpu_uuid in uuid_to_index, f"compute app refers to unknown GPU UUID {gpu_uuid}"
    index = uuid_to_index[gpu_uuid]
    assert index not in apps, f"physical GPU {index} has more than one compute application"
    pid = int(pid_text)
    assert pid != 1, "PID 1 must never be a burn worker"
    assert pid not in all_pids, f"PID {pid} appears on more than one physical GPU"
    args = cmdline(pid)
    burn_script = os.path.join(project_dir, "scripts", "gpu_burn.py")
    assert any(arg == burn_script for arg in args), (
        f"GPU {index} PID {pid} is not the expected project burn: {args}"
    )
    ppid = parent_pid(pid)
    apps[index] = {
        "pid": pid,
        "ppid": ppid,
        "process_name": process_name,
        "used_memory": float(used_memory_text),
        "args": args,
    }
    all_pids.add(pid)

assert set(apps) == expected_indices, f"not every physical GPU has one burn: {sorted(apps)}"
parents = {app["ppid"] for app in apps.values()}
assert len(parents) == 1, f"burn workers do not share one supervisor: {sorted(parents)}"
supervisor_pid = next(iter(parents))
supervisor_args = cmdline(supervisor_pid)
assert len(supervisor_args) >= 2, f"unexpected supervisor cmdline: {supervisor_args}"
assert os.path.basename(supervisor_args[0]) in {"bash", "sh"}, (
    f"unexpected supervisor executable: {supervisor_args}"
)
assert supervisor_args[1] == runtime, (
    f"burn supervisor is not the expected isolated workflow: {supervisor_args}"
)

for slot in range(8):
    ready_log = Path(f"/tmp/project_gpu_burn_four_model_workflow_gpu{slot}.log")
    assert ready_log.is_file(), f"missing burn readiness log: {ready_log}"
    text = ready_log.read_text(errors="replace")
    assert "gpu_burn_ready" in text, f"readiness marker missing from {ready_log}"

samples = {index: [] for index in expected_indices}
print("=== repeated utilization samples (physical GPU index: utilization %, memory MiB, power W) ===")
for sample_number in range(1, 6):
    sample_rows = rows(run(
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,power.draw",
        "--format=csv,noheader,nounits",
    ))
    assert len(sample_rows) == 8, f"sample {sample_number}: expected 8 rows, got {sample_rows}"
    current = {}
    for row in sample_rows:
        index = int(row[0])
        current[index] = (float(row[1]), float(row[2]), float(row[3]))
    assert set(current) == expected_indices, f"sample {sample_number}: bad indices {sorted(current)}"
    print(f"sample {sample_number}: " + " | ".join(
        f"GPU {index}: {current[index][0]:.0f}%, {current[index][1]:.0f} MiB, {current[index][2]:.0f} W"
        for index in sorted(current)
    ))
    for index, values in current.items():
        samples[index].append(values)
    if sample_number < 5:
        time.sleep(2)

print("=== exact physical-GPU-to-burn mapping ===")
for index in sorted(gpus):
    app = apps[index]
    util_values = [sample[0] for sample in samples[index]]
    memory_values = [sample[1] for sample in samples[index]]
    assert max(util_values) >= 90, f"GPU {index} never reached 90% utilization: {util_values}"
    assert statistics.mean(util_values) >= 75, f"GPU {index} mean utilization below 75%: {util_values}"
    assert min(memory_values) >= 1800, f"GPU {index} burn memory unexpectedly low: {memory_values}"
    print(
        f"GPU {index} | {gpus[index]['name']} | {gpus[index]['uuid']} | "
        f"PID {app['pid']} | PPID {app['ppid']} | app_memory {app['used_memory']:.0f} MiB | "
        f"util_mean {statistics.mean(util_values):.1f}% | cmdline {' '.join(app['args'])}"
    )

print(f"common supervisor PID {supervisor_pid}: {' '.join(supervisor_args)}")
print("ALL_EIGHT_B200_GPUS_HAVE_EXACTLY_ONE_SUPERVISED_PROJECT_BURN")
print("NO_OTHER_GPU_COMPUTE_PROCESSES")
print("TH2 EIGHT-GPU BURN AUDIT COMPLETE; PROCESSES UNMODIFIED")
PY

echo '=== final nvidia-smi snapshot (read-only) ==='
nvidia-smi
