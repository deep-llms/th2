#0
#th2-idle-after-nested-watcher-isolation-audit-20260829-a01
set -uo pipefail

TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_LOG_DIR="$TASK_OUTPUT_BASE/logs/nested_ladder_phase1_20260829"
TASK_LOCK_FILE="$TASK_OUTPUT_BASE/.locks/nested_phase1_then_burn.lock"
TASK_TRAIN_LOG="$TASK_LOG_DIR/nested_ladder_tied_t4.log"

echo '=== identity ==='
date -u
hostname

echo '=== training runner ==='
pgrep -af '[r]un_experiments.py' || true
pgrep -af '[t]rain_compositional.py' | head -20 || true

echo '=== tmux sessions and panes ==='
tmux list-sessions 2>&1 || true
tmux list-panes -a -F 'session=#{session_name} pane_pid=#{pane_pid} command=#{pane_current_command} dead=#{pane_dead} status=#{pane_dead_status}' 2>&1 || true

echo '=== watcher-like processes ==='
ps -eo pid,ppid,sid,pgid,stat,etime,args \
    | grep -E 'nested.phase|commands\.sh|sleep 30|gpu_burn' \
    | grep -v grep || true

echo '=== lock-file and kernel lock records ==='
ls -li "$TASK_LOCK_FILE" 2>&1 || true
TASK_LOCK_INODE="$(stat -c %i "$TASK_LOCK_FILE" 2>/dev/null || true)"
echo "lock_inode=$TASK_LOCK_INODE"
if [ -n "$TASK_LOCK_INODE" ]; then
    awk -v inode="$TASK_LOCK_INODE" '$0 ~ (":" inode "($| )") {print}' /proc/locks || true
fi
command -v lslocks || true
lslocks -n -o PID,COMMAND,TYPE,MODE,PATH 2>&1 \
    | grep -F 'nested_phase1_then_burn.lock' || true

echo '=== scan every visible process FD for the watcher lock ==='
TASK_FOUND=0
for TASK_FD_DIR in /proc/[0-9]*/fd; do
    TASK_PID="${TASK_FD_DIR#/proc/}"
    TASK_PID="${TASK_PID%%/*}"
    for TASK_FD in "$TASK_FD_DIR"/*; do
        TASK_TARGET="$(readlink "$TASK_FD" 2>/dev/null || true)"
        if [ "$TASK_TARGET" = "$TASK_LOCK_FILE" ]; then
            echo "lock_holder_pid=$TASK_PID fd=${TASK_FD##*/} cmdline=$(tr '\0' ' ' < "/proc/$TASK_PID/cmdline")"
            TASK_FOUND=$((TASK_FOUND + 1))
        fi
    done
done
echo "visible_lock_holder_count=$TASK_FOUND"

echo '=== burn and GPU state ==='
pgrep -af '[g]pu_burn.py' || echo 'gpu_burn_process=absent'
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,power.draw \
    --format=csv,noheader

echo '=== recent training metrics ==='
tail -c 150000 "$TASK_TRAIN_LOG" | tr '\r' '\n' \
    | grep -E "\{'loss':" | tail -10 || true
echo 'TH2 WATCHER PROCESS INSPECTION COMPLETE'
