#1 +180+a
#th2-replace-project-burns-with-runner-burn-all-eight-20260830-a01
set -euo pipefail

TASK_OLD_PROJECT_BURN_PATH=/mnt/local/@PROJECT@/scripts/gpu_burn.py
TASK_OLD_RUNTIME=/tmp/run_phase1_four_model_eval_finetune_burn_2949bab98ea9.sh
TASK_RUNNER_BURN=/tmp/llm_pretrain_burn.py
TASK_BURN_PYTHON=/usr/bin/python3
TASK_BURN_LOG=/tmp/llm_pretrain_burn_all_gpus.log
TASK_BURN_PID_FILE=/tmp/llm_pretrain_burn_launcher.pid

gpu_pids() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | sed '/^[[:space:]]*$/d;s/[[:space:]]//g' | sort -nu
}

cmdline() {
    tr '\0' ' ' < "/proc/$1/cmdline"
}

echo '=== verify the current eight project burns before cancelling ==='
date -u
hostname
test -s "$TASK_RUNNER_BURN"
test -x "$TASK_BURN_PYTHON"
mapfile -t TASK_OLD_PIDS < <(gpu_pids)
[[ "${#TASK_OLD_PIDS[@]}" -eq 8 ]] || {
    echo "REFUSE: expected exactly 8 current GPU PIDs, found ${#TASK_OLD_PIDS[@]}" >&2
    exit 1
}

declare -A TASK_OLD_PID_SET=()
TASK_OLD_SUPERVISOR=
for TASK_PID in "${TASK_OLD_PIDS[@]}"; do
    [[ "$TASK_PID" != 1 ]]
    TASK_CMDLINE="$(cmdline "$TASK_PID")"
    [[ "$TASK_CMDLINE" == *"$TASK_OLD_PROJECT_BURN_PATH"* ]] || {
        echo "REFUSE: unexpected GPU PID $TASK_PID: $TASK_CMDLINE" >&2
        exit 1
    }
    TASK_PPID="$(awk '/^PPid:/ {print $2}' "/proc/$TASK_PID/status")"
    [[ -n "$TASK_PPID" && "$TASK_PPID" != 1 ]]
    if [[ -z "$TASK_OLD_SUPERVISOR" ]]; then
        TASK_OLD_SUPERVISOR="$TASK_PPID"
    else
        [[ "$TASK_PPID" == "$TASK_OLD_SUPERVISOR" ]] || {
            echo 'REFUSE: current project burns do not share one supervisor' >&2
            exit 1
        }
    fi
    TASK_OLD_PID_SET["$TASK_PID"]=1
    echo "verified_old_burn pid=$TASK_PID ppid=$TASK_PPID cmdline=$TASK_CMDLINE"
done

TASK_SUPERVISOR_CMDLINE="$(cmdline "$TASK_OLD_SUPERVISOR")"
[[ "$TASK_SUPERVISOR_CMDLINE" == "bash $TASK_OLD_RUNTIME " ]] || {
    echo "REFUSE: unexpected old supervisor $TASK_OLD_SUPERVISOR: $TASK_SUPERVISOR_CMDLINE" >&2
    exit 1
}
echo "verified_old_supervisor pid=$TASK_OLD_SUPERVISOR cmdline=$TASK_SUPERVISOR_CMDLINE"

echo '=== cancel only the verified current project burns ==='
kill -TERM "${TASK_OLD_PIDS[@]}"
for _ in $(seq 1 60); do
    mapfile -t TASK_REMAINING_GPU_PIDS < <(gpu_pids)
    [[ "${#TASK_REMAINING_GPU_PIDS[@]}" -eq 0 ]] && break
    sleep 1
done
mapfile -t TASK_REMAINING_GPU_PIDS < <(gpu_pids)
if [[ "${#TASK_REMAINING_GPU_PIDS[@]}" -ne 0 ]]; then
    for TASK_PID in "${TASK_REMAINING_GPU_PIDS[@]}"; do
        [[ -n "${TASK_OLD_PID_SET[$TASK_PID]:-}" ]] || {
            echo "REFUSE: unexpected GPU PID appeared during cancellation: $TASK_PID" >&2
            exit 1
        }
        [[ "$(cmdline "$TASK_PID")" == *"$TASK_OLD_PROJECT_BURN_PATH"* ]] || {
            echo "REFUSE: PID identity changed during cancellation: $TASK_PID" >&2
            exit 1
        }
    done
    kill -KILL "${TASK_REMAINING_GPU_PIDS[@]}"
fi

for _ in $(seq 1 30); do
    if ! kill -0 "$TASK_OLD_SUPERVISOR" 2>/dev/null; then
        break
    fi
    sleep 1
done
if kill -0 "$TASK_OLD_SUPERVISOR" 2>/dev/null; then
    [[ "$(cmdline "$TASK_OLD_SUPERVISOR")" == "bash $TASK_OLD_RUNTIME " ]] || {
        echo 'REFUSE: old supervisor PID identity changed' >&2
        exit 1
    }
    kill -TERM "$TASK_OLD_SUPERVISOR"
fi
sleep 3
mapfile -t TASK_AFTER_CANCEL_PIDS < <(gpu_pids)
[[ "${#TASK_AFTER_CANCEL_PIDS[@]}" -eq 0 ]] || {
    echo "ERROR: GPUs not free after cancellation: ${TASK_AFTER_CANCEL_PIDS[*]}" >&2
    exit 1
}
echo 'OLD PROJECT WORKFLOW CANCELLED; ALL GPUS FREE'
nvidia-smi

echo '=== launch the runner-provided pretraining burn on all eight GPUs ==='
rm -f "$TASK_BURN_LOG" "$TASK_BURN_PID_FILE"
nohup env \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    MASTER_ADDR=127.0.0.1 \
    MASTER_PORT=29500 \
    "$TASK_BURN_PYTHON" -u "$TASK_RUNNER_BURN" \
    >"$TASK_BURN_LOG" 2>&1 &
TASK_NEW_LAUNCHER_PID=$!
printf '%s\n' "$TASK_NEW_LAUNCHER_PID" > "$TASK_BURN_PID_FILE"
echo "runner_burn_launcher_pid=$TASK_NEW_LAUNCHER_PID log=$TASK_BURN_LOG"

sleep 45
kill -0 "$TASK_NEW_LAUNCHER_PID"
grep -Fq 'GPU burn: 8 visible GPU(s)' "$TASK_BURN_LOG"
cat "$TASK_BURN_LOG"

echo '=== verify exactly one runner-burn descendant per physical B200 ==='
"$TASK_BURN_PYTHON" - "$TASK_NEW_LAUNCHER_PID" <<'PY'
import csv
from pathlib import Path
import statistics
import subprocess
import sys
import time

launcher = int(sys.argv[1])


def run(*args):
    return subprocess.check_output(args, text=True).strip()


def rows(output):
    if not output:
        return []
    return [[value.strip() for value in row] for row in csv.reader(output.splitlines())]


def ppid(pid):
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("PPid:"):
            return int(line.split()[1])
    raise RuntimeError(f"PPid missing for {pid}")


def descends_from(pid, ancestor):
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == ancestor:
            return True
        seen.add(pid)
        pid = ppid(pid)
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
assert len(app_rows) == 8, f"expected 8 compute workers, found {app_rows}"
apps = {}
for uuid, pid_text, process_name, memory_text in app_rows:
    assert uuid in uuid_to_index, uuid
    index = uuid_to_index[uuid]
    assert index not in apps, f"more than one process on GPU {index}"
    pid = int(pid_text)
    assert descends_from(pid, launcher), (
        f"GPU {index} PID {pid} is not descended from launcher {launcher}"
    )
    apps[index] = (pid, process_name, float(memory_text))
assert set(apps) == set(range(8)), apps

samples = {index: [] for index in range(8)}
for sample_number in range(1, 6):
    sample_rows = rows(run(
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,power.draw",
        "--format=csv,noheader,nounits",
    ))
    assert len(sample_rows) == 8, sample_rows
    current = {int(row[0]): tuple(float(value) for value in row[1:]) for row in sample_rows}
    assert set(current) == set(range(8)), current
    print(f"sample {sample_number}: " + " | ".join(
        f"GPU {index}: {current[index][0]:.0f}%, {current[index][1]:.0f} MiB, {current[index][2]:.0f} W"
        for index in range(8)
    ))
    for index in range(8):
        samples[index].append(current[index])
    if sample_number < 5:
        time.sleep(2)

for index in range(8):
    utilization = [sample[0] for sample in samples[index]]
    assert max(utilization) >= 90, (index, utilization)
    assert statistics.mean(utilization) >= 75, (index, utilization)
    pid, process_name, app_memory = apps[index]
    print(
        f"GPU {index} uuid={gpus[index]} pid={pid} process={process_name} "
        f"app_memory={app_memory:.0f}MiB util_mean={statistics.mean(utilization):.1f}%"
    )

print("ALL_EIGHT_B200_GPUS_RUNNING_RUNNER_PRETRAIN_BURN")
print("NO_OTHER_GPU_COMPUTE_PROCESSES")
PY

nvidia-smi
echo 'TH2 CORRECT RUNNER GPU BURN ACTIVE ON ALL 8 GPUS'
