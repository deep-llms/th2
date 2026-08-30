#1 +30+a
#th2-check-live-runner-burn-memory-all-eight-20260830-a01
set -euo pipefail

TASK_BURN_SCRIPT=/tmp/llm_pretrain_burn.py
TASK_BURN_PID_FILE=/tmp/llm_pretrain_burn_launcher.pid
test -s "$TASK_BURN_SCRIPT"
test -s "$TASK_BURN_PID_FILE"
TASK_LAUNCHER_PID="$(cat "$TASK_BURN_PID_FILE")"
[[ "$TASK_LAUNCHER_PID" =~ ^[0-9]+$ ]]
[[ "$TASK_LAUNCHER_PID" != 1 ]]
kill -0 "$TASK_LAUNCHER_PID"
TASK_LAUNCHER_CMDLINE="$(tr '\0' ' ' < "/proc/$TASK_LAUNCHER_PID/cmdline")"
[[ "$TASK_LAUNCHER_CMDLINE" == *"$TASK_BURN_SCRIPT"* ]]

echo "runner_burn_launcher_pid=$TASK_LAUNCHER_PID cmdline=$TASK_LAUNCHER_CMDLINE"
echo '=== three live per-GPU samples: index, UUID, name, memory MiB, utilization %, power W ==='
for TASK_SAMPLE in 1 2 3; do
    echo "sample=$TASK_SAMPLE"
    nvidia-smi \
        --query-gpu=index,uuid,name,memory.used,utilization.gpu,power.draw \
        --format=csv,noheader,nounits
    [[ "$TASK_SAMPLE" -eq 3 ]] || sleep 2
done

echo '=== GPU compute applications: UUID, PID, process, memory MiB ==='
nvidia-smi \
    --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits

/usr/bin/python3 - "$TASK_LAUNCHER_PID" <<'PY'
import csv
from pathlib import Path
import subprocess
import sys

launcher = int(sys.argv[1])


def rows(command):
    output = subprocess.check_output(command, text=True).strip()
    return [[field.strip() for field in row] for row in csv.reader(output.splitlines())] if output else []


def parent(pid):
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("PPid:"):
            return int(line.split()[1])
    raise RuntimeError(f"missing PPid for PID {pid}")


def descendant(pid):
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == launcher:
            return True
        seen.add(pid)
        pid = parent(pid)
    return False


gpu_rows = rows([
    "nvidia-smi",
    "--query-gpu=index,uuid,name,memory.used,utilization.gpu,power.draw",
    "--format=csv,noheader,nounits",
])
assert len(gpu_rows) == 8, gpu_rows
uuid_to_index = {}
for index_text, uuid, name, memory, utilization, power in gpu_rows:
    index = int(index_text)
    assert index in range(8) and index not in uuid_to_index.values()
    assert "B200" in name, (index, name)
    uuid_to_index[uuid] = index

app_rows = rows([
    "nvidia-smi",
    "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
    "--format=csv,noheader,nounits",
])
assert len(app_rows) == 8, app_rows
seen_indices = set()
for uuid, pid_text, process_name, memory in app_rows:
    assert uuid in uuid_to_index, uuid
    index = uuid_to_index[uuid]
    assert index not in seen_indices, f"multiple compute apps on GPU {index}"
    pid = int(pid_text)
    assert descendant(pid), f"GPU {index} PID {pid} is not a descendant of launcher {launcher}"
    seen_indices.add(index)
assert seen_indices == set(range(8)), seen_indices
print("ALL_EIGHT_GPUS_HAVE_EXACTLY_ONE_RUNNER_BURN_WORKER")
print("NO_OTHER_GPU_COMPUTE_PROCESSES")
PY

echo 'TH2 LIVE RUNNER BURN MEMORY CHECK COMPLETE; PROCESSES UNMODIFIED'
