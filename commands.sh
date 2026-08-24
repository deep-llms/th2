#1 +120+a
#th2-check-shared-local-full-workflow-20260824-2
#!/usr/bin/env bash
set -euo pipefail

TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_TRAIN_OUTPUT="$TASK_OUTPUT_BASE/shared_local_tied_g16"
TASK_CONTINUATION_LOG="$TASK_OUTPUT_BASE/logs/shared_local_tied_g16/shared_local_tied_g16_resume_10000_to_full.log"
TASK_EVAL_LOG="$TASK_TRAIN_OUTPUT/eval_parallel_full.log"
TASK_FINETUNE_OUTPUT="$TASK_OUTPUT_BASE/finetune_shared_local_tied_g16_full"

echo '=== th2 SharedLocal full workflow status ==='
date -u
hostname

echo '=== active workflow processes ==='
pgrep -af 'train_compositional.py|accelerate.*launch|eval_parallel.py|eval_checkpoint.py|finetune/(run_all|train).py' \
    || echo 'NO MATCHING TRAIN/EVAL/FINETUNE PROCESS'

echo '=== GPU state ==='
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu \
    --format=csv,noheader
echo '=== GPU compute processes ==='
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader \
    || true

echo '=== latest training checkpoint ==='
latest_checkpoint=$(find "$TASK_TRAIN_OUTPUT" -mindepth 1 -maxdepth 1 -type d \
    -name 'checkpoint-*' -printf '%f\n' 2>/dev/null \
    | sed 's/checkpoint-//' | sort -n | tail -1)
if [[ -n "$latest_checkpoint" ]]; then
    echo "LATEST_CHECKPOINT=$latest_checkpoint"
else
    echo 'LATEST_CHECKPOINT=none'
fi

echo '=== workflow artifacts ==='
for path in \
    "$TASK_TRAIN_OUTPUT/trainer_state.json" \
    "$TASK_TRAIN_OUTPUT/model.safetensors" \
    "$TASK_TRAIN_OUTPUT/eval_ppl.json" \
    "$TASK_TRAIN_OUTPUT/eval_benchmarks.json" \
    "$TASK_EVAL_LOG" \
    "$TASK_FINETUNE_OUTPUT/summary.md"; do
    if [[ -e "$path" ]]; then
        stat -c 'PRESENT %y %s %n' "$path"
    else
        echo "PENDING $path"
    fi
done
if [[ -d "$TASK_FINETUNE_OUTPUT" ]]; then
    echo "FINETUNE_JSON_COUNT=$(find "$TASK_FINETUNE_OUTPUT" -maxdepth 1 -type f -name '*.json' | wc -l)"
fi

echo '=== latest continuation metrics ==='
if [[ -s "$TASK_CONTINUATION_LOG" ]]; then
    tr '\r' '\n' < "$TASK_CONTINUATION_LOG" \
        | grep -E "\{'loss':|Training complete\. Model saved to:|global_step|FULL_TRAINING_STATE_OK" \
        | tail -20 \
        || true
else
    echo "MISSING $TASK_CONTINUATION_LOG"
fi

echo '=== current stage log tail ==='
if [[ -s "$TASK_EVAL_LOG" ]]; then
    tail -80 "$TASK_EVAL_LOG"
elif [[ -s "$TASK_CONTINUATION_LOG" ]]; then
    tr '\r' '\n' < "$TASK_CONTINUATION_LOG" | tail -80
else
    echo 'NO WORKFLOW LOG FOUND'
fi

echo '=== narrow fatal-signature scan ==='
TASK_FATAL_PATTERN='Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|ChildFailedError|ProcessExitedException|NCCL.*(unhandled|system error|remote process exited|watchdog|timeout)|Segmentation fault|Bus error'
TASK_FATAL_FOUND=0
for log_path in "$TASK_CONTINUATION_LOG" "$TASK_EVAL_LOG"; do
    if [[ -s "$log_path" ]] && grep -niE "$TASK_FATAL_PATTERN" "$log_path"; then
        TASK_FATAL_FOUND=1
    fi
done
if [[ -d "$TASK_FINETUNE_OUTPUT" ]]; then
    while IFS= read -r log_path; do
        if grep -niE "$TASK_FATAL_PATTERN|eval failed:|FAILED \(code" "$log_path"; then
            TASK_FATAL_FOUND=1
        fi
    done < <(find "$TASK_FINETUNE_OUTPUT" -maxdepth 1 -type f -name '*.log' -print)
fi
[[ "$TASK_FATAL_FOUND" -eq 0 ]] || {
    echo 'FATAL SIGNATURE FOUND; ACTIVE WORKFLOW WAS NOT MODIFIED'
    exit 1
}
echo 'NO FATAL SIGNATURE FOUND'
echo 'TH2 SHARED LOCAL FULL WORKFLOW STATUS CHECK COMPLETE'
