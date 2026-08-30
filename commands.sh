#1 +240+a
#th2-replace-runner-burn-with-165g-all-eight-20260830-a01
set -euo pipefail

TASK_SOURCE=/mnt/local/@PROJECT@/resources/llm_pretrain_burn.py
TASK_TARGET=/tmp/llm_pretrain_burn.py
TASK_PYTHON=/usr/bin/python3
TASK_LOG=/tmp/llm_pretrain_burn_all_gpus.log
TASK_PID_FILE=/tmp/llm_pretrain_burn_launcher.pid
TASK_EXPECTED_SHA=97d96734d14f1dba578208a929f822e2693770f5f320daae2fbcff87853260aa

gpu_pids() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | sed '/^[[:space:]]*$/d;s/[[:space:]]//g' | sort -nu
}

echo '=== verify current runner-burn process tree ==='
date -u
hostname
test -s "$TASK_SOURCE"
test -s "$TASK_TARGET"
test -x "$TASK_PYTHON"
test -s "$TASK_PID_FILE"
echo "$TASK_EXPECTED_SHA  $TASK_SOURCE" | sha256sum -c -

TASK_OLD_LAUNCHER="$(cat "$TASK_PID_FILE")"
[[ "$TASK_OLD_LAUNCHER" =~ ^[0-9]+$ ]]
[[ "$TASK_OLD_LAUNCHER" != 1 ]]
kill -0 "$TASK_OLD_LAUNCHER"
TASK_OLD_LAUNCHER_CMD="$(tr '\0' ' ' < "/proc/$TASK_OLD_LAUNCHER/cmdline")"
[[ "$TASK_OLD_LAUNCHER_CMD" == *"$TASK_TARGET"* ]] || {
    echo "REFUSE: unexpected launcher $TASK_OLD_LAUNCHER: $TASK_OLD_LAUNCHER_CMD" >&2
    exit 1
}

mapfile -t TASK_OLD_GPU_PIDS < <(gpu_pids)
[[ "${#TASK_OLD_GPU_PIDS[@]}" -eq 8 ]] || {
    echo "REFUSE: expected 8 current runner-burn GPU workers, found ${#TASK_OLD_GPU_PIDS[@]}" >&2
    exit 1
}
"$TASK_PYTHON" - "$TASK_OLD_LAUNCHER" "${TASK_OLD_GPU_PIDS[@]}" <<'PY'
from pathlib import Path
import sys

launcher = int(sys.argv[1])
pids = [int(value) for value in sys.argv[2:]]


def parent(pid):
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("PPid:"):
            return int(line.split()[1])
    raise RuntimeError(f"missing PPid for {pid}")


def descendant(pid):
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == launcher:
            return True
        seen.add(pid)
        pid = parent(pid)
    return False


assert len(pids) == 8 and len(set(pids)) == 8
for pid in pids:
    assert pid != 1 and descendant(pid), (pid, launcher)
print(f"VERIFIED_CURRENT_RUNNER_BURN launcher={launcher} gpu_pids={pids}")
PY

echo '=== cancel only the verified current runner-burn tree ==='
kill -TERM "${TASK_OLD_GPU_PIDS[@]}" "$TASK_OLD_LAUNCHER" 2>/dev/null || true
for _ in $(seq 1 60); do
    mapfile -t TASK_REMAINING_GPU_PIDS < <(gpu_pids)
    [[ "${#TASK_REMAINING_GPU_PIDS[@]}" -eq 0 ]] && break
    sleep 1
done
mapfile -t TASK_REMAINING_GPU_PIDS < <(gpu_pids)
if [[ "${#TASK_REMAINING_GPU_PIDS[@]}" -ne 0 ]]; then
    declare -A TASK_OLD_GPU_PID_SET=()
    for TASK_PID in "${TASK_OLD_GPU_PIDS[@]}"; do
        TASK_OLD_GPU_PID_SET["$TASK_PID"]=1
    done
    for TASK_PID in "${TASK_REMAINING_GPU_PIDS[@]}"; do
        [[ -n "${TASK_OLD_GPU_PID_SET[$TASK_PID]:-}" ]] || {
            echo "REFUSE: unexpected GPU PID appeared during cancellation: $TASK_PID" >&2
            exit 1
        }
    done
    kill -KILL "${TASK_REMAINING_GPU_PIDS[@]}"
fi
if kill -0 "$TASK_OLD_LAUNCHER" 2>/dev/null; then
    [[ "$(tr '\0' ' ' < "/proc/$TASK_OLD_LAUNCHER/cmdline")" == *"$TASK_TARGET"* ]] || {
        echo 'REFUSE: old launcher PID identity changed' >&2
        exit 1
    }
    kill -KILL "$TASK_OLD_LAUNCHER"
fi
sleep 5
mapfile -t TASK_AFTER_CANCEL_PIDS < <(gpu_pids)
[[ "${#TASK_AFTER_CANCEL_PIDS[@]}" -eq 0 ]] || {
    echo "ERROR: GPUs not free after cancellation: ${TASK_AFTER_CANCEL_PIDS[*]}" >&2
    exit 1
}
echo 'OLD RUNNER BURN CANCELLED; ALL GPUS FREE'
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits

echo '=== install and launch the 165 GiB runner burn ==='
install -m 0644 "$TASK_SOURCE" "$TASK_TARGET"
echo "$TASK_EXPECTED_SHA  $TASK_TARGET" | sha256sum -c -
rm -f "$TASK_LOG" "$TASK_PID_FILE"
nohup env \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    MASTER_ADDR=127.0.0.1 \
    MASTER_PORT=29500 \
    GPU_BURN_TARGET_GIB=165 \
    GPU_BURN_MIN_FREE_GIB=8 \
    GPU_BURN_MATRIX_SIZE=8192 \
    NCCL_DEBUG=WARN \
    "$TASK_PYTHON" -u "$TASK_TARGET" \
    >"$TASK_LOG" 2>&1 &
TASK_NEW_LAUNCHER=$!
printf '%s\n' "$TASK_NEW_LAUNCHER" > "$TASK_PID_FILE"
echo "new_launcher_pid=$TASK_NEW_LAUNCHER log=$TASK_LOG"

TASK_READY=0
for _ in $(seq 1 240); do
    if ! kill -0 "$TASK_NEW_LAUNCHER" 2>/dev/null; then
        echo 'ERROR: 165 GiB burn launcher exited during startup' >&2
        cat "$TASK_LOG" >&2
        exit 1
    fi
    TASK_READY_COUNT="$(grep -Fc 'gpu_burn_ready' "$TASK_LOG" 2>/dev/null || true)"
    mapfile -t TASK_NEW_GPU_PIDS < <(gpu_pids)
    if [[ "$TASK_READY_COUNT" -eq 8 && "${#TASK_NEW_GPU_PIDS[@]}" -eq 8 ]]; then
        TASK_READY=1
        break
    fi
    sleep 1
done
[[ "$TASK_READY" -eq 1 ]] || {
    echo 'ERROR: 165 GiB burn did not become ready on all 8 GPUs' >&2
    cat "$TASK_LOG" >&2
    exit 1
}
cat "$TASK_LOG"

echo '=== verify 160-170 GiB and sustained compute on every physical GPU ==='
"$TASK_PYTHON" - "$TASK_NEW_LAUNCHER" <<'PY'
import csv
from pathlib import Path
import statistics
import subprocess
import sys
import time

launcher = int(sys.argv[1])
minimum_memory_mib = 160 * 1024
maximum_memory_mib = 170 * 1024


def run(*args):
    return subprocess.check_output(args, text=True).strip()


def rows(output):
    if not output:
        return []
    return [[field.strip() for field in row] for row in csv.reader(output.splitlines())]


def parent(pid):
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("PPid:"):
            return int(line.split()[1])
    raise RuntimeError(f"missing PPid for {pid}")


def descendant(pid):
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == launcher:
            return True
        seen.add(pid)
        pid = parent(pid)
    return False


gpu_rows = rows(run(
    "nvidia-smi",
    "--query-gpu=index,uuid,name",
    "--format=csv,noheader,nounits",
))
assert len(gpu_rows) == 8, gpu_rows
gpus = {}
for index_text, uuid, name in gpu_rows:
    index = int(index_text)
    assert index in range(8) and index not in gpus
    assert "B200" in name, (index, name)
    gpus[index] = uuid
assert set(gpus) == set(range(8))
uuid_to_index = {uuid: index for index, uuid in gpus.items()}

app_rows = rows(run(
    "nvidia-smi",
    "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
    "--format=csv,noheader,nounits",
))
assert len(app_rows) == 8, app_rows
apps = {}
for uuid, pid_text, process_name, memory_text in app_rows:
    assert uuid in uuid_to_index, uuid
    index = uuid_to_index[uuid]
    assert index not in apps, f"multiple compute apps on GPU {index}"
    pid = int(pid_text)
    memory_mib = float(memory_text)
    assert descendant(pid), f"GPU {index} PID {pid} is not descended from {launcher}"
    apps[index] = (pid, process_name, memory_mib)
assert set(apps) == set(range(8)), apps

samples = {index: [] for index in range(8)}
for sample_number in range(1, 6):
    sample_rows = rows(run(
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ))
    assert len(sample_rows) == 8, sample_rows
    current = {int(row[0]): tuple(float(value) for value in row[1:]) for row in sample_rows}
    assert set(current) == set(range(8)), current
    print(f"sample {sample_number}: " + " | ".join(
        f"GPU {index}: {current[index][0]:.0f} MiB, {current[index][1]:.0f}%, {current[index][2]:.0f} W"
        for index in range(8)
    ))
    for index in range(8):
        samples[index].append(current[index])
    if sample_number < 5:
        time.sleep(2)

for index in range(8):
    memory_values = [sample[0] for sample in samples[index]]
    utilization_values = [sample[1] for sample in samples[index]]
    assert min(memory_values) >= minimum_memory_mib, (index, memory_values)
    assert max(memory_values) <= maximum_memory_mib, (index, memory_values)
    assert max(utilization_values) >= 90, (index, utilization_values)
    assert statistics.mean(utilization_values) >= 75, (index, utilization_values)
    pid, process_name, app_memory = apps[index]
    print(
        f"GPU {index} uuid={gpus[index]} pid={pid} process={process_name} "
        f"app_memory={app_memory:.0f}MiB "
        f"memory_mean={statistics.mean(memory_values):.0f}MiB "
        f"util_mean={statistics.mean(utilization_values):.1f}%"
    )

print("ALL_EIGHT_B200_GPUS_RUNNING_165_GIB_PRETRAIN_BURN")
print("EVERY_GPU_MEMORY_BETWEEN_160_AND_170_GIB")
print("NO_OTHER_GPU_COMPUTE_PROCESSES")
PY

nvidia-smi
echo 'TH2 165 GIB GPU BURN ACTIVE AND VERIFIED ON ALL 8 GPUS'
