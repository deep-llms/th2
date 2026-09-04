#1 +300+a
#th2-cancel-five-checkpoint-finetune-and-restore-burn-20260904-a01
set -euo pipefail

TASK_PROJECT_DIR="${SPARSE_EMB_PROJECT_DIR:-/mnt/local/@PROJECT@}"
TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_FINETUNE_OUTPUT="$TASK_OUTPUT_BASE/finetune_ranklift_hashedv2_btmos_steps_6500_10000_20260904_a01"
TASK_BURN_SOURCE="$TASK_PROJECT_DIR/resources/llm_pretrain_burn.py"
TASK_BURN_TARGET=/tmp/llm_pretrain_burn.py
TASK_BURN_LOG=/tmp/llm_pretrain_burn_all_gpus.log
TASK_BURN_PID_FILE=/tmp/llm_pretrain_burn_launcher.pid
TASK_BURN_PYTHON=/usr/bin/python3

gpu_pids() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | sed '/^[[:space:]]*$/d;s/[[:space:]]//g' | sort -nu
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

burn_launcher() {
    "$TASK_BURN_PYTHON" - "$TASK_BURN_TARGET" <<'PY'
from pathlib import Path
import subprocess
import sys

burn_path = sys.argv[1]
raw = subprocess.check_output(
    ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
    text=True,
)
pids = sorted({int(line.strip()) for line in raw.splitlines() if line.strip()})
assert len(pids) == 8, pids

def cmdline(pid):
    return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')

def parent(pid):
    for line in Path(f'/proc/{pid}/status').read_text().splitlines():
        if line.startswith('PPid:'):
            return int(line.split()[1])
    raise RuntimeError(pid)

launcher_sets = []
for pid in pids:
    assert pid != 1
    cursor = pid
    seen = set()
    launchers = set()
    while cursor > 1 and cursor not in seen:
        seen.add(cursor)
        if burn_path in cmdline(cursor):
            launchers.add(cursor)
        cursor = parent(cursor)
    assert launchers, (pid, cmdline(pid))
    launcher_sets.append(launchers)
common = set.intersection(*launcher_sets)
assert len(common) == 1, (common, launcher_sets)
launcher = next(iter(common))
assert launcher != 1
print(launcher)
PY
}

verify_burn() {
    local launcher
    launcher="$(burn_launcher)" || return 1
    [[ "$launcher" =~ ^[0-9]+$ && "$launcher" -ne 1 ]] || return 1
    [[ "$(grep -Fc 'gpu_burn_ready' "$TASK_BURN_LOG" 2>/dev/null || true)" -eq 8 ]] || return 1
    [[ "$(grep -Fc 'world_size=8' "$TASK_BURN_LOG" 2>/dev/null || true)" -eq 8 ]] || return 1
    grep -Fq 'gpu_burn_progress' "$TASK_BURN_LOG" || return 1
    "$TASK_BURN_PYTHON" - <<'PY'
import csv
import subprocess

raw = subprocess.check_output(
    ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total',
     '--format=csv,noheader,nounits'], text=True,
)
rows = [[field.strip() for field in row] for row in csv.reader(raw.splitlines()) if row]
assert len(rows) == 8, rows
for index, name, used, total in rows:
    assert 'B200' in name, (index, name)
    fraction = float(used) / float(total)
    assert 0.80 <= fraction <= 0.90, (index, used, total, fraction)
print('BURN_MEMORY_OK gpus=8 target_fraction=0.85')
PY
    echo "BURN_VERIFIED launcher=$launcher world_size=8"
}

echo '=== cancellation preflight ==='
date -u
hostname
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader
mapfile -t TASK_INITIAL_GPU_PIDS < <(gpu_pids)
echo "initial_gpu_pids=${TASK_INITIAL_GPU_PIDS[*]:-none}"

if [[ "${#TASK_INITIAL_GPU_PIDS[@]}" -eq 8 ]] && verify_burn; then
    echo 'FINETUNING_ALREADY_ENDED_AND_CORRECT_BURN_ALREADY_ACTIVE'
elif [[ "${#TASK_INITIAL_GPU_PIDS[@]}" -gt 0 ]]; then
    "$TASK_BURN_PYTHON" - "$TASK_FINETUNE_OUTPUT" "${TASK_INITIAL_GPU_PIDS[@]}" <<'PY'
from pathlib import Path
import sys

output_dir = sys.argv[1]
pids = [int(value) for value in sys.argv[2:]]
assert 1 <= len(pids) <= 8, pids
for pid in pids:
    assert pid != 1
    cmd = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
    assert 'finetune/train.py' in cmd, (pid, cmd)
    assert output_dir in cmd, (pid, cmd)
    print(f'VERIFIED_FINETUNE_GPU_PID pid={pid} cmd={cmd}')
PY
    echo '=== signal only verified finetuning GPU compute PIDs ==='
    kill -9 "${TASK_INITIAL_GPU_PIDS[@]}" 2>/dev/null || true
    sleep 5
    mapfile -t TASK_LINGERING_FINETUNE_PIDS < <(
        pgrep -f '[f]inetune/train.py' | sort -nu || true
    )
    if [[ "${#TASK_LINGERING_FINETUNE_PIDS[@]}" -gt 0 ]]; then
        "$TASK_BURN_PYTHON" - "$TASK_FINETUNE_OUTPUT" \
            "${TASK_LINGERING_FINETUNE_PIDS[@]}" <<'PY'
from pathlib import Path
import sys

output_dir = sys.argv[1]
for value in sys.argv[2:]:
    pid = int(value)
    assert pid != 1
    cmd = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
    assert 'finetune/train.py' in cmd, (pid, cmd)
    assert output_dir in cmd, (pid, cmd)
    print(f'VERIFIED_LINGERING_FINETUNE_PID pid={pid} cmd={cmd}')
PY
        kill -9 "${TASK_LINGERING_FINETUNE_PIDS[@]}" 2>/dev/null || true
    fi
else
    echo 'NO_GPU_COMPUTE_PROCESSES_AT_PREFLIGHT'
fi

echo '=== wait for finetune coordinator/workflow shutdown or recovery burn ==='
TASK_RECOVERY_READY=0
for _ in $(seq 1 180); do
    mapfile -t TASK_NOW_GPU_PIDS < <(gpu_pids)
    if [[ "${#TASK_NOW_GPU_PIDS[@]}" -eq 8 ]] && verify_burn >/dev/null 2>&1; then
        TASK_RECOVERY_READY=1
        echo 'WORKFLOW_RECOVERY_BURN_READY'
        break
    fi
    if [[ "${#TASK_NOW_GPU_PIDS[@]}" -eq 0 ]] \
            && ! pgrep -f '[f]inetune/run_all.py' >/dev/null \
            && ! pgrep -f '[r]un_five_checkpoint_eval_finetune_burn.sh' >/dev/null; then
        echo 'FINETUNE_AND_WORKFLOW_PROCESSES_EXITED'
        break
    fi
    sleep 1
done

if [[ "$TASK_RECOVERY_READY" -ne 1 ]]; then
    mapfile -t TASK_BEFORE_BURN_PIDS < <(gpu_pids)
    [[ "${#TASK_BEFORE_BURN_PIDS[@]}" -eq 0 ]] \
        || die "unexpected GPU PIDs remain before burn: ${TASK_BEFORE_BURN_PIDS[*]}"
    pgrep -af '[f]inetune/run_all.py' && die 'finetune coordinator still exists'
    pgrep -af '[f]inetune/train.py' && die 'finetune worker processes still exist'
    pgrep -af '[r]un_five_checkpoint_eval_finetune_burn.sh' && die 'workflow process still exists'

    echo '=== install and launch current communicating high-memory burn ==='
    test -s "$TASK_BURN_SOURCE"
    install -m 0644 "$TASK_BURN_SOURCE" "$TASK_BURN_TARGET"
    cmp "$TASK_BURN_SOURCE" "$TASK_BURN_TARGET"
    rm -f "$TASK_BURN_LOG" "$TASK_BURN_PID_FILE"
    nohup env \
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        MASTER_ADDR=127.0.0.1 \
        MASTER_PORT=29500 \
        NCCL_DEBUG=WARN \
        GPU_BURN_MIN_WORLD_SIZE=8 \
        "$TASK_BURN_PYTHON" -u "$TASK_BURN_TARGET" \
        >"$TASK_BURN_LOG" 2>&1 &
    TASK_LAUNCHER=$!
    [[ "$TASK_LAUNCHER" -ne 1 ]]
    printf '%s\n' "$TASK_LAUNCHER" > "$TASK_BURN_PID_FILE"

    TASK_READY=0
    for _ in $(seq 1 300); do
        kill -0 "$TASK_LAUNCHER" 2>/dev/null \
            || { cat "$TASK_BURN_LOG"; die 'burn launcher exited during startup'; }
        if [[ "$(grep -Fc 'gpu_burn_ready' "$TASK_BURN_LOG" 2>/dev/null || true)" -eq 8 ]] \
                && [[ "$(gpu_pids | wc -l)" -eq 8 ]]; then
            TASK_READY=1
            break
        fi
        sleep 1
    done
    [[ "$TASK_READY" -eq 1 ]] || { cat "$TASK_BURN_LOG"; die 'burn did not become ready'; }
    for _ in $(seq 1 180); do
        grep -Fq 'gpu_burn_progress' "$TASK_BURN_LOG" && break
        sleep 1
    done
fi

echo '=== final verified state ==='
verify_burn || { cat "$TASK_BURN_LOG"; die 'final burn verification failed'; }
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader
echo 'TH2 FINETUNING CANCELLED; CORRECT 8-GPU COMMUNICATING HIGH-MEMORY BURN ACTIVE'
