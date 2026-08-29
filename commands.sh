#1 +30+a
#th2-check-dense-tied-ddp-default-training-20260829-a01
set -u

TASK_OUTPUT=/mnt/local/_outputs/@PROJECT@/dense_tied_baseline_b200_ddp_default
TASK_LOG_DIR=/mnt/local/_outputs/@PROJECT@/logs/dense_tied_baseline_b200_ddp_default_20260829
TASK_TRAIN_LOG="$TASK_LOG_DIR/dense_tied_baseline_b200_ddp_default.log"

echo '=== timestamp and host ==='
date -u
hostname

echo '=== workflow and training processes ==='
pgrep -af '[r]un_dense_ddp_default_then_burn.sh' || echo 'workflow_process=absent'
pgrep -af '[r]un_experiments.py' || echo 'run_experiments_process=absent'
pgrep -af '[t]rain.py' || echo 'train_process=absent'

echo '=== output and latest checkpoints ==='
if [ -d "$TASK_OUTPUT" ]; then
    du -sh "$TASK_OUTPUT"
    find "$TASK_OUTPUT" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' \
        -printf '%f\n' | sort -V | tail -10
else
    echo 'training_output=absent'
fi

echo '=== recent training metrics and fatal scan ==='
if [ -s "$TASK_TRAIN_LOG" ]; then
    tail -c 200000 "$TASK_TRAIN_LOG" | tr '\r' '\n' \
        | grep -E "\{'loss':|find_unused_parameters=True|Traceback|CUDA out of memory|OutOfMemoryError|NCCL.*(unhandled|system error|watchdog|timeout)" \
        | tail -30 || true
    echo "training_log_bytes=$(stat -c %s "$TASK_TRAIN_LOG")"
else
    echo 'training_log=absent_or_empty'
fi

echo '=== GPU state ==='
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,power.draw \
    --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits || true

echo 'TH2 READ-ONLY DDP-DEFAULT TRAINING STATUS COMPLETE'
