#1 +30+a
#th2-audit-nested-burn-watcher-isolation-20260829-a01
set -euo pipefail

TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_LOG_DIR="$TASK_OUTPUT_BASE/logs/nested_ladder_phase1_20260829"
TASK_EXPERIMENT_LOG="$TASK_LOG_DIR/experiments.log"
TASK_TRAIN_LOG="$TASK_LOG_DIR/nested_ladder_tied_t4.log"
TASK_LOCK_FILE="$TASK_OUTPUT_BASE/.locks/nested_phase1_then_burn.lock"
TASK_EXPECTED_RUNNER_FRAGMENT="run_experiments.py --experiments 21 22 --stop-at-step 10000 --log-dir $TASK_LOG_DIR"

echo '=== identity ==='
date -u
hostname

echo '=== resolve exact training runner ==='
TASK_RUNNER_PIDS=()
while read -r TASK_PID; do
    [ -n "$TASK_PID" ] || continue
    TASK_CMDLINE="$(tr '\0' ' ' < "/proc/$TASK_PID/cmdline")"
    if [[ "$TASK_CMDLINE" == *"$TASK_EXPECTED_RUNNER_FRAGMENT"* ]]; then
        TASK_RUNNER_PIDS+=("$TASK_PID")
        echo "runner_pid=$TASK_PID cmdline=$TASK_CMDLINE"
    fi
done < <(pgrep -f '[r]un_experiments.py' || true)
test "${#TASK_RUNNER_PIDS[@]}" -eq 1
TASK_RUNNER_PID="${TASK_RUNNER_PIDS[0]}"

echo '=== resolve independent watcher through its held lock ==='
test -e "$TASK_LOCK_FILE"
mapfile -t TASK_LOCK_PIDS < <(
    fuser "$TASK_LOCK_FILE" 2>/dev/null \
        | tr ' ' '\n' | sed '/^[[:space:]]*$/d' | sort -nu
)
test "${#TASK_LOCK_PIDS[@]}" -eq 1
TASK_WATCHER_PID="${TASK_LOCK_PIDS[0]}"
test "$TASK_WATCHER_PID" -ne "$TASK_RUNNER_PID"
TASK_WATCHER_CMDLINE="$(tr '\0' ' ' < "/proc/$TASK_WATCHER_PID/cmdline")"
echo "watcher_pid=$TASK_WATCHER_PID cmdline=$TASK_WATCHER_CMDLINE"
echo "watcher_lock_fd9=$(readlink "/proc/$TASK_WATCHER_PID/fd/9")"
test "$(readlink "/proc/$TASK_WATCHER_PID/fd/9")" = "$TASK_LOCK_FILE"

echo '=== prove separate process/session ownership ==='
ps -o pid,ppid,sid,pgid,stat,etime,cmd \
    -p "$TASK_RUNNER_PID,$TASK_WATCHER_PID"
TASK_RUNNER_SID="$(ps -o sid= -p "$TASK_RUNNER_PID" | tr -d ' ')"
TASK_WATCHER_SID="$(ps -o sid= -p "$TASK_WATCHER_PID" | tr -d ' ')"
test -n "$TASK_RUNNER_SID"
test -n "$TASK_WATCHER_SID"
test "$TASK_RUNNER_SID" != "$TASK_WATCHER_SID"
echo "runner_sid=$TASK_RUNNER_SID watcher_sid=$TASK_WATCHER_SID isolated=true"
echo 'watcher children:'
ps -o pid,ppid,sid,stat,etime,cmd --ppid "$TASK_WATCHER_PID" || true

echo '=== confirm watcher is still in read-only wait stage ==='
grep -Fq '[1/2] nested_ladder_tied_t4' "$TASK_EXPERIMENT_LOG"
if grep -Fq 'groupreduce_matched_nested_tied_t4: STOPPED at step 10000' \
        "$TASK_EXPERIMENT_LOG"; then
    echo 'second_experiment_already_complete=true'
else
    echo 'second_experiment_complete=false; watcher_must_remain_waiting'
fi
if pgrep -af '[g]pu_burn.py'; then
    echo 'ERROR: burn process exists before both experiments completed' >&2
    exit 1
fi

echo '=== training remains healthy ==='
tail -20 "$TASK_EXPERIMENT_LOG"
if grep -HniE 'CUDA out of memory|OutOfMemoryError|NCCL.*(unhandled|system error|remote process exited|watchdog|timeout)|Segmentation fault|Bus error' \
        "$TASK_TRAIN_LOG"; then
    echo 'ERROR: fatal training signature detected' >&2
    exit 1
fi
tail -c 180000 "$TASK_TRAIN_LOG" | tr '\r' '\n' \
    | grep -E "\{'loss':" | tail -10 || true

mapfile -t TASK_GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -nu
)
test "${#TASK_GPU_PIDS[@]}" -eq 8
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,power.draw \
    --format=csv,noheader
echo 'TH2 WATCHER ISOLATION AUDIT OK; CURRENT TRAINING UNTOUCHED'
