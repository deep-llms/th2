#1 +30+a
#th2-readonly-verify-active-eight-gpu-nccl-communication-20260830-a01
set -euo pipefail

TASK_SCRIPT=/tmp/llm_pretrain_burn.py
TASK_LOG=/tmp/llm_pretrain_burn_all_gpus.log
TASK_PID_FILE=/tmp/llm_pretrain_burn_launcher.pid
TASK_EXPECTED_SHA=97d96734d14f1dba578208a929f822e2693770f5f320daae2fbcff87853260aa

echo '=== script/process identity and launch environment ==='
date -u
hostname
test -s "$TASK_SCRIPT"
test -s "$TASK_LOG"
test -s "$TASK_PID_FILE"
echo "$TASK_EXPECTED_SHA  $TASK_SCRIPT" | sha256sum -c -
TASK_LAUNCHER_PID="$(cat "$TASK_PID_FILE")"
[[ "$TASK_LAUNCHER_PID" =~ ^[0-9]+$ ]]
[[ "$TASK_LAUNCHER_PID" != 1 ]]
test -d "/proc/$TASK_LAUNCHER_PID"
tr '\0' ' ' < "/proc/$TASK_LAUNCHER_PID/cmdline"
echo
tr '\0' '\n' < "/proc/$TASK_LAUNCHER_PID/environ" \
    | grep -E '^(CUDA_VISIBLE_DEVICES|MASTER_ADDR|MASTER_PORT|GPU_BURN_TARGET_GIB|GPU_BURN_MATRIX_SIZE)='

echo '=== static control-flow proof and completed warm-up collective proof ==='
/usr/bin/python3 - "$TASK_SCRIPT" "$TASK_LOG" <<'PY'
import ast
from pathlib import Path
import re
import sys

script = Path(sys.argv[1])
log = Path(sys.argv[2])
source = script.read_text()
tree = ast.parse(source)

burn = next(
    node for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "burn"
)


def call_name(call):
    parts = []
    node = call.func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


calls = [node for node in ast.walk(burn) if isinstance(node, ast.Call)]
init_calls = [node for node in calls if call_name(node) == "dist.init_process_group"]
all_reduce_calls = [node for node in calls if call_name(node) == "dist.all_reduce"]
spawn_calls = [
    node for node in ast.walk(tree)
    if isinstance(node, ast.Call) and call_name(node) == "mp.spawn"
]
assert len(init_calls) == 1, [node.lineno for node in init_calls]
assert len(all_reduce_calls) == 2, [node.lineno for node in all_reduce_calls]
assert len(spawn_calls) == 1, [node.lineno for node in spawn_calls]

init = init_calls[0]
backend = next(
    keyword.value.value for keyword in init.keywords
    if keyword.arg == "backend" and isinstance(keyword.value, ast.Constant)
)
assert backend == "nccl", backend

while_nodes = [node for node in ast.walk(burn) if isinstance(node, ast.While)]
assert len(while_nodes) == 1
loop = while_nodes[0]
assert isinstance(loop.test, ast.Constant) and loop.test.value is True
loop_all_reduces = [
    node for node in ast.walk(loop)
    if isinstance(node, ast.Call) and call_name(node) == "dist.all_reduce"
]
loop_mm = [
    node for node in ast.walk(loop)
    if isinstance(node, ast.Call) and call_name(node) == "torch.mm"
]
loop_sync = [
    node for node in ast.walk(loop)
    if isinstance(node, ast.Call) and call_name(node) == "torch.cuda.synchronize"
]
assert len(loop_mm) == len(loop_all_reduces) == len(loop_sync) == 1
assert loop_mm[0].lineno < loop_all_reduces[0].lineno < loop_sync[0].lineno

log_text = log.read_text(errors="replace")
ready_ranks = [int(value) for value in re.findall(r"gpu_burn_ready rank=(\d+)", log_text)]
assert sorted(ready_ranks) == list(range(8)), ready_ranks
assert "NCCL version" in log_text

print(f"NCCL init line: {init.lineno}")
print(f"warm-up all-reduce line: {min(node.lineno for node in all_reduce_calls)}")
print(f"loop GEMM/all-reduce/synchronize lines: "
      f"{loop_mm[0].lineno}/{loop_all_reduces[0].lineno}/{loop_sync[0].lineno}")
print(f"completed warm-up collective ranks: {sorted(ready_ranks)}")
print("STATIC_AND_READINESS_PROOF_OF_ONE_EIGHT_RANK_NCCL_WORLD")
PY

echo '=== exact live worker ancestry and ongoing loop progress evidence ==='
/usr/bin/python3 - "$TASK_LAUNCHER_PID" <<'PY'
import csv
from pathlib import Path
import statistics
import subprocess
import sys
import time

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
    "--query-gpu=index,uuid,name,memory.total,memory.used",
    "--format=csv,noheader,nounits",
])
assert len(gpu_rows) == 8, gpu_rows
gpus = {}
for index_text, uuid, name, total_text, used_text in gpu_rows:
    index = int(index_text)
    assert index in range(8) and index not in gpus
    assert "B200" in name, (index, name)
    gpus[index] = (uuid, float(total_text), float(used_text))
assert set(gpus) == set(range(8))
uuid_to_index = {values[0]: index for index, values in gpus.items()}

app_rows = rows([
    "nvidia-smi",
    "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
    "--format=csv,noheader,nounits",
])
assert len(app_rows) == 8, app_rows
apps = {}
for uuid, pid_text, process_name, memory_text in app_rows:
    assert uuid in uuid_to_index, uuid
    index = uuid_to_index[uuid]
    assert index not in apps
    pid = int(pid_text)
    assert descendant(pid), (index, pid, launcher)
    apps[index] = (pid, process_name, float(memory_text))
assert set(apps) == set(range(8))

samples = {index: [] for index in range(8)}
for sample_number in range(1, 6):
    sample_rows = rows([
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ])
    assert len(sample_rows) == 8
    current = {int(row[0]): tuple(float(value) for value in row[1:]) for row in sample_rows}
    assert set(current) == set(range(8))
    print(f"sample {sample_number}: " + " | ".join(
        f"GPU {index}: {current[index][0]:.0f}MiB {current[index][1]:.0f}% {current[index][2]:.0f}W"
        for index in range(8)
    ))
    for index in range(8):
        samples[index].append(current[index])
    if sample_number < 5:
        time.sleep(2)

for index in range(8):
    memory = [sample[0] for sample in samples[index]]
    utilization = [sample[1] for sample in samples[index]]
    assert min(memory) >= 160 * 1024, (index, memory)
    assert statistics.mean(utilization) >= 90, (index, utilization)
    pid, process_name, process_memory = apps[index]
    print(
        f"GPU {index} pid={pid} process={process_name} process_memory={process_memory:.0f}MiB "
        f"util_mean={statistics.mean(utilization):.1f}%"
    )

print("EIGHT_LIVE_DESCENDANT_RANKS_CONTINUE_HIGH_UTILIZATION_GEMM_COLLECTIVE_LOOP")
print("NO_OTHER_GPU_COMPUTE_PROCESSES")
PY

echo '=== B200 NVLink topology and active-link status (read-only) ==='
nvidia-smi topo -m
nvidia-smi nvlink --status || true

echo '=== NVLink throughput counter snapshots when supported (read-only) ==='
for TASK_COUNTER in 0 1 2 3; do
    echo "counter=$TASK_COUNTER snapshot=1"
    nvidia-smi nvlink --getthroughput "$TASK_COUNTER" || true
    sleep 2
    echo "counter=$TASK_COUNTER snapshot=2"
    nvidia-smi nvlink --getthroughput "$TASK_COUNTER" || true
done

echo 'READ_ONLY EIGHT-GPU NCCL COMMUNICATION VERIFICATION COMPLETE; PROCESSES UNMODIFIED'
