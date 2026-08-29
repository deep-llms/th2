#0
#th2-idle-after-nested-ladder-phase1-start-20260829-a01
set -uo pipefail

TASK_OUTPUT=/mnt/local/_outputs/@PROJECT@/nested_ladder_tied_t4
TASK_LOG_DIR=/mnt/local/_outputs/@PROJECT@/logs/nested_ladder_phase1_20260829
TASK_EXPERIMENT_LOG="$TASK_LOG_DIR/experiments.log"
TASK_TRAIN_LOG="$TASK_LOG_DIR/nested_ladder_tied_t4.log"

echo '=== identity ==='
date -u
hostname

echo '=== workflow processes ==='
pgrep -af '[r]un_experiments.py' || {
    echo 'ERROR: sequential runner is absent' >&2
    exit 1
}
pgrep -af '[a]ccelerate.commands.launch|[a]ccelerate launch' || {
    echo 'ERROR: Accelerate launcher is absent' >&2
    exit 1
}
mapfile -t TASK_TRAIN_PIDS < <(pgrep -f '[t]rain_compositional.py' | sort -nu)
test "${#TASK_TRAIN_PIDS[@]}" -ge 8
echo "train_related_process_count=${#TASK_TRAIN_PIDS[@]}"

echo '=== output and logs ==='
if [ -d "$TASK_OUTPUT" ]; then
    du -sh "$TASK_OUTPUT"
    find "$TASK_OUTPUT" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' \
        -printf '%f\n' | sort -V | tail -10 || true
else
    echo 'training_output=not_created_yet'
fi
if [ -s "$TASK_EXPERIMENT_LOG" ]; then
    tail -40 "$TASK_EXPERIMENT_LOG"
else
    echo 'experiment_log=not_created_yet'
fi
if [ -s "$TASK_TRAIN_LOG" ]; then
    if grep -HniE 'CUDA out of memory|OutOfMemoryError|NCCL.*(unhandled|system error|remote process exited|watchdog|timeout)|Segmentation fault|Bus error' \
            "$TASK_TRAIN_LOG" "$TASK_EXPERIMENT_LOG"; then
        echo 'ERROR: fatal signature found in current training logs' >&2
        exit 1
    fi
    tail -c 200000 "$TASK_TRAIN_LOG" | tr '\r' '\n' \
        | grep -E "Embedding:|Total parameters:|Trainable parameters:|\{'loss':" \
        | tail -30 || true
else
    echo 'training_log=not_created_yet'
fi

echo '=== live GPU state ==='
mapfile -t TASK_GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -nu
)
echo "gpu_compute_pid_count=${#TASK_GPU_PIDS[@]}"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,power.draw \
    --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits
echo 'TH2 NESTED LADDER PHASE1 TRAINING IS LIVE'
