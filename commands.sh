#1 +30+a
#th2-readonly-verify-165g-burn-script-memory-and-nccl-20260830-a01
set -euo pipefail

TASK_SOURCE=/mnt/local/@PROJECT@/resources/llm_pretrain_burn.py
TASK_TARGET=/tmp/llm_pretrain_burn.py
TASK_LOG=/tmp/llm_pretrain_burn_all_gpus.log
TASK_PID_FILE=/tmp/llm_pretrain_burn_launcher.pid
TASK_EXPECTED_SHA=97d96734d14f1dba578208a929f822e2693770f5f320daae2fbcff87853260aa

echo '=== exact remote /tmp burn script and identity ==='
date -u
hostname
test -s "$TASK_SOURCE"
test -s "$TASK_TARGET"
test -s "$TASK_LOG"
test -s "$TASK_PID_FILE"
echo "$TASK_EXPECTED_SHA  $TASK_SOURCE" | sha256sum -c -
echo "$TASK_EXPECTED_SHA  $TASK_TARGET" | sha256sum -c -
cmp "$TASK_SOURCE" "$TASK_TARGET"
cat "$TASK_TARGET"

echo '=== static communication and persistent-memory checks ==='
grep -Fq 'dist.init_process_group(backend="nccl", rank=rank, world_size=world)' "$TASK_TARGET"
grep -Fq 'dist.all_reduce(token, op=dist.ReduceOp.SUM)' "$TASK_TARGET"
grep -Fq 'mp.spawn(burn, args=(world,), nprocs=world)' "$TASK_TARGET"
grep -Fq 'memory_reserve = torch.empty' "$TASK_TARGET"
grep -Fq 'memory_reserve.zero_()' "$TASK_TARGET"
grep -Fq 'target_used_gib=165.00' "$TASK_LOG"
[[ "$(grep -Fc 'gpu_burn_ready' "$TASK_LOG")" -eq 8 ]]

TASK_LAUNCHER_PID="$(cat "$TASK_PID_FILE")"
[[ "$TASK_LAUNCHER_PID" =~ ^[0-9]+$ ]]
[[ "$TASK_LAUNCHER_PID" != 1 ]]
test -d "/proc/$TASK_LAUNCHER_PID"
TASK_LAUNCHER_CMDLINE="$(tr '\0' ' ' < "/proc/$TASK_LAUNCHER_PID/cmdline")"
[[ "$TASK_LAUNCHER_CMDLINE" == *"$TASK_TARGET"* ]]
echo "launcher_pid=$TASK_LAUNCHER_PID cmdline=$TASK_LAUNCHER_CMDLINE"

echo '=== five live memory samples: index, UUID, total MiB, used MiB, free MiB, utilization %, power W ==='
for TASK_SAMPLE in 1 2 3 4 5; do
    echo "sample=$TASK_SAMPLE"
    nvidia-smi \
        --query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu,power.draw \
        --format=csv,noheader,nounits
    [[ "$TASK_SAMPLE" -eq 5 ]] || sleep 2
done

echo '=== exact compute applications ==='
nvidia-smi \
    --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits

/usr/bin/python3 - "$TASK_LAUNCHER_PID" <<'PY'
import csv
from pathlib import Path
import statistics
import subprocess
import sys

launcher = int(sys.argv[1])
lower_memory_mib = 160 * 1024
upper_memory_mib = 170 * 1024


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
    "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,power.draw",
    "--format=csv,noheader,nounits",
])
assert len(gpu_rows) == 8, gpu_rows
gpus = {}
for row in gpu_rows:
    index = int(row[0])
    uuid, name = row[1], row[2]
    total, used, free, utilization, power = map(float, row[3:])
    assert index in range(8) and index not in gpus
    assert "B200" in name, (index, name)
    assert lower_memory_mib <= used <= upper_memory_mib, (index, used)
    assert abs(total - used - free) <= 2, (index, total, used, free)
    gpus[index] = {
        "uuid": uuid,
        "total": total,
        "used": used,
        "free": free,
        "utilization": utilization,
        "power": power,
    }
assert set(gpus) == set(range(8)), gpus

uuid_to_index = {gpu["uuid"]: index for index, gpu in gpus.items()}
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
    assert index not in apps, f"multiple GPU processes on GPU {index}"
    pid = int(pid_text)
    assert descendant(pid), f"GPU {index} PID {pid} does not descend from launcher {launcher}"
    apps[index] = (pid, process_name, float(memory_text))
assert set(apps) == set(range(8)), apps

for index in range(8):
    gpu = gpus[index]
    pid, process_name, process_memory = apps[index]
    used_percent = 100.0 * gpu["used"] / gpu["total"]
    assert used_percent >= 90.0, (index, used_percent)
    print(
        f"GPU {index} uuid={gpu['uuid']} pid={pid} process={process_name} "
        f"process_memory={process_memory:.0f}MiB used={gpu['used']:.0f}MiB "
        f"total={gpu['total']:.0f}MiB used_percent={used_percent:.2f}% "
        f"free={gpu['free']:.0f}MiB utilization={gpu['utilization']:.0f}% "
        f"power={gpu['power']:.0f}W"
    )

print("REMOTE_TMP_SCRIPT_BYTE_IDENTICAL_TO_COMMITTED_165_GIB_SCRIPT")
print("NCCL_WORLD_INITIALIZATION_AND_EVERY_ITERATION_ALL_REDUCE_PRESENT")
print("ALL_EIGHT_GPUS_USE_MORE_THAN_90_PERCENT_OF_HBM")
print("EXACTLY_ONE_BURN_WORKER_PER_GPU_AND_NO_OTHER_GPU_COMPUTE_PROCESSES")
print("READ_ONLY_VERIFICATION_COMPLETE; PROCESSES_UNMODIFIED")
PY
