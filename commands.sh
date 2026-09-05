#1 +180+a
#th2-readonly-inspect-tiered-stop-and-control-progress-20260905-a04
set -euo pipefail

TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_TIERED="$TASK_OUTPUT_BASE/tiered_ranklift_lb_t4_c512"
TASK_CONTROL="$TASK_OUTPUT_BASE/groupreduce_matched_lb_t4"
TASK_LOG_DIR="$TASK_OUTPUT_BASE/logs/tiered_c512_lb_groupreduce_10k_20260904_a01"
TASK_MARKER="$TASK_OUTPUT_BASE/status/tiered_c512_lb_groupreduce_10k_20260904_a01.complete"

echo '=== read-only current state ==='
date -u
nvidia-smi \
    --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader
nvidia-smi \
    --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits || true
pgrep -af '[r]un_experiments.py|[a]ccelerate.commands.launch|[t]rain_compositional.py|[l]lm_pretrain_burn.py' || true

echo '=== completion marker ==='
if [[ -s "$TASK_MARKER" ]]; then
    cat "$TASK_MARKER"
else
    echo 'COMPLETION_MARKER_NOT_PRESENT'
fi

echo '=== exact runner summary ==='
cat "$TASK_LOG_DIR/experiments.log"

for TASK_NAME in tiered_ranklift_lb_t4_c512 groupreduce_matched_lb_t4; do
    TASK_OUTPUT="$TASK_OUTPUT_BASE/$TASK_NAME"
    TASK_LOG="$TASK_LOG_DIR/$TASK_NAME.log"
    echo "=== $TASK_NAME checkpoint state ==="
    find "$TASK_OUTPUT" -mindepth 1 -maxdepth 1 -type d \
        -name 'checkpoint-*' -printf '%f\n' 2>/dev/null | sort -V | tail -12 || true
    echo "=== $TASK_NAME traceback/fatal context ==="
    if [[ -s "$TASK_LOG" ]]; then
        grep -n -E -B 15 -A 25 \
            'Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|ChildFailedError|ProcessExitedException|Segmentation fault|Bus error' \
            "$TASK_LOG" || echo 'NO_FATAL_PATTERN'
        echo "=== $TASK_NAME latest log ==="
        tail -100 "$TASK_LOG"
    else
        echo 'LOG_NOT_CREATED'
    fi
done

echo 'TH2 TIERED STOP/CONTROL PROGRESS READ-ONLY INSPECTION COMPLETE'
