#1 +30+a
#th2-check-sampling-to-dense-baseline-status-20260829-a01
set -u

TASK_OUTPUT=/mnt/local/_outputs/@PROJECT@/dense_tied_baseline_b200
TASK_LOG_DIR=/mnt/local/_outputs/@PROJECT@/logs/dense_tied_baseline_b200_fresh_b200_20260828
TASK_DATASET=/mnt/local/_data/@PROJECT@/data/Qwen_Qwen3-0.6B

echo '=== timestamp and host ==='
date -u
hostname

echo '=== sampler process ==='
pgrep -af '[p]repare_data.py sample' || echo 'sampler_process=absent'

echo '=== handoff process ==='
pgrep -af '[w]atch_sampling_then_train_dense_baseline.sh' || echo 'handoff_process=absent'

echo '=== baseline processes ==='
pgrep -af '[r]un_experiments.py' || echo 'run_experiments_process=absent'
pgrep -af '[t]rain.py' || echo 'train_process=absent'

echo '=== sampled outputs ==='
du -sh "$TASK_DATASET" 2>/dev/null || echo 'sampled_dataset=absent'
for TASK_LANG in en vi zh ru de ar; do
    TASK_TRAIN_SHARDS="$(find "$TASK_DATASET/train/$TASK_LANG" -mindepth 1 -maxdepth 1 -type d -name 'shard_*' 2>/dev/null | wc -l)"
    if [ -d "$TASK_DATASET/eval/$TASK_LANG" ]; then
        TASK_EVAL_STATE=present
    else
        TASK_EVAL_STATE=absent
    fi
    echo "lang=$TASK_LANG train_shards=$TASK_TRAIN_SHARDS eval=$TASK_EVAL_STATE"
done

echo '=== baseline output and checkpoints ==='
if [ -d "$TASK_OUTPUT" ]; then
    du -sh "$TASK_OUTPUT"
    find "$TASK_OUTPUT" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' \
        -printf '%f\n' | sort -V | tail -10
else
    echo 'baseline_output=absent'
fi

echo '=== baseline logs ==='
if [ -d "$TASK_LOG_DIR" ]; then
    find "$TASK_LOG_DIR" -maxdepth 1 -type f -printf '%f %s_bytes\n' | sort
    tail -40 "$TASK_LOG_DIR/experiments.log" 2>/dev/null || true
else
    echo 'baseline_log_dir=absent'
fi

echo '=== GPU state ==='
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,power.draw \
    --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits || true

echo 'TH2 READ-ONLY BASELINE HANDOFF STATUS COMPLETE'
