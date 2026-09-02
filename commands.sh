#1 +60+a
#th2-readonly-check-culturax-sampling-progress-20260902-a21
set -euo pipefail

echo '=== CulturaX sampling status ==='
date -u
hostname

TASK_OUTPUT_ROOT=/mnt/local/_data/@PROJECT@/data/Qwen_Qwen3-0.6B

echo '=== matching sampling processes ==='
mapfile -t TASK_SAMPLE_PIDS < <(
    ps -eo pid=,comm=,args= |
        awk '$2 ~ /^python([0-9.]*)?$/ && $0 ~ /prepare_data\.py sample/ {print $1}'
)
echo "sampling_process_count=${#TASK_SAMPLE_PIDS[@]}"
for TASK_PID in "${TASK_SAMPLE_PIDS[@]}"; do
    ps -p "$TASK_PID" -o pid=,etimes=,stat=,args=
done

echo '=== sampled output inventory ==='
if [ -d "$TASK_OUTPUT_ROOT" ]; then
    du -sh "$TASK_OUTPUT_ROOT"
else
    echo 'sampled_output_root=ABSENT'
fi

TASK_COMPLETE=1
for TASK_LANG in en vi zh ru de ar; do
    TASK_TRAIN_DIR="$TASK_OUTPUT_ROOT/train/$TASK_LANG"
    TASK_EVAL_DIR="$TASK_OUTPUT_ROOT/eval/$TASK_LANG"
    TASK_SHARD_COUNT=0
    TASK_TRAIN_ARROW_COUNT=0
    TASK_EVAL_ARROW_COUNT=0
    if [ -d "$TASK_TRAIN_DIR" ]; then
        TASK_SHARD_COUNT="$(find "$TASK_TRAIN_DIR" -mindepth 1 -maxdepth 1 -type d -name 'shard_*' | wc -l)"
        TASK_TRAIN_ARROW_COUNT="$(find "$TASK_TRAIN_DIR" -type f -name '*.arrow' | wc -l)"
    fi
    if [ -d "$TASK_EVAL_DIR" ]; then
        TASK_EVAL_ARROW_COUNT="$(find "$TASK_EVAL_DIR" -maxdepth 1 -type f -name '*.arrow' | wc -l)"
    fi
    printf '%s train_shards=%s train_arrow_files=%s eval_arrow_files=%s\n' \
        "$TASK_LANG" "$TASK_SHARD_COUNT" "$TASK_TRAIN_ARROW_COUNT" "$TASK_EVAL_ARROW_COUNT"
    if [ "$TASK_SHARD_COUNT" -eq 0 ] || [ "$TASK_TRAIN_ARROW_COUNT" -eq 0 ] || \
       [ "$TASK_EVAL_ARROW_COUNT" -eq 0 ] || [ ! -s "$TASK_EVAL_DIR/dataset_info.json" ] || \
       [ ! -s "$TASK_EVAL_DIR/state.json" ]; then
        TASK_COMPLETE=0
    fi
done

if [ "$TASK_COMPLETE" -eq 1 ] && [ "${#TASK_SAMPLE_PIDS[@]}" -eq 0 ]; then
    echo 'SAMPLING_STATE=COMPLETE'
elif [ "${#TASK_SAMPLE_PIDS[@]}" -gt 0 ]; then
    echo 'SAMPLING_STATE=RUNNING'
else
    echo 'SAMPLING_STATE=STOPPED_INCOMPLETE'
fi

echo '=== GPU state (read only) ==='
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader,nounits
