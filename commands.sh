#1 +60+a
#th2-inspect-independent-lr-final-traceback-context-20260823
set -euo pipefail

TASK_LOG_DIR=/mnt/local/_outputs/deep-llms_th2/logs/lowrank_independent_output_r128
TASK_EXPERIMENT_LOG="$TASK_LOG_DIR/experiments.log"
TASK_TRAIN_LOG="$TASK_LOG_DIR/lowrank_independent_output_r128.log"

echo '=== timestamp and GPUs ==='
date -u
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true

echo '=== runner summary ==='
cat "$TASK_EXPERIMENT_LOG"

echo '=== fatal-signature line numbers ==='
grep -n -E -i 'Traceback|CUDA out of memory|OutOfMemoryError|RuntimeError:|SignalException|SIGTERM|NCCL[^[:cntrl:]]*(unhandled|system error|remote process exited|watchdog|collective operation timeout)' "$TASK_TRAIN_LOG" || true

echo '=== traceback context ==='
grep -n -B 30 -A 60 -E -i 'Traceback|CUDA out of memory|OutOfMemoryError|RuntimeError:|SignalException|SIGTERM|NCCL[^[:cntrl:]]*(unhandled|system error|remote process exited|watchdog|collective operation timeout)' "$TASK_TRAIN_LOG" | tail -400 || true

echo '=== final training-log tail ==='
tail -160 "$TASK_TRAIN_LOG"
echo 'TH2 FINAL TRACEBACK CONTEXT INSPECTION COMPLETE'
