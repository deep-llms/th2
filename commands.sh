#1 +240+a
#th2-readonly-check-five-checkpoint-eval-finetune-completion-20260904-a01
set -u

TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_FINETUNE_OUTPUT="$TASK_OUTPUT_BASE/finetune_ranklift_hashedv2_btmos_steps_6500_10000_20260904_a01"
TASK_COMPLETION_MARKER="$TASK_OUTPUT_BASE/eval_finetune_ranklift_hashedv2_btmos_steps_6500_10000_20260904_a01.complete"
TASK_EVAL_LAUNCH_LOG="$TASK_OUTPUT_BASE/eval_parallel_ranklift_hashedv2_btmos_steps_6500_10000_20260904_a01.log"
TASK_CHECKPOINTS=(
    "$TASK_OUTPUT_BASE/ranklift_tied_c124_m460/checkpoint-6500"
    "$TASK_OUTPUT_BASE/ranklift_tied_c124_m460/checkpoint-10000"
    "$TASK_OUTPUT_BASE/product_code_quota_h6144/checkpoint-6500"
    "$TASK_OUTPUT_BASE/product_code_quota_h6144/checkpoint-10000"
    "$TASK_OUTPUT_BASE/btmos_k3_c256_lb/checkpoint-6500"
)

echo '=== time and completion marker ==='
date -u
hostname
if [[ -s "$TASK_COMPLETION_MARKER" ]]; then
    echo 'COMPLETION_MARKER_PRESENT'
    cat "$TASK_COMPLETION_MARKER"
else
    echo 'COMPLETION_MARKER_ABSENT'
fi

echo '=== workflow processes ==='
pgrep -af '[r]un_five_checkpoint_eval_finetune_burn' || true
pgrep -af '[e]val/eval_parallel.py' || true
pgrep -af '[e]val/eval_checkpoint.py' || true
pgrep -af '[f]inetune/run_all.py' || true
pgrep -af '[f]inetune/train.py' || true

echo '=== GPU state ==='
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits || true

echo '=== eval artifacts ==='
for TASK_CHECKPOINT in "${TASK_CHECKPOINTS[@]}"; do
    echo "checkpoint=$TASK_CHECKPOINT"
    for TASK_NAME in eval.log eval_ppl.json eval_benchmarks.json; do
        TASK_PATH="$TASK_CHECKPOINT/$TASK_NAME"
        if [[ -s "$TASK_PATH" ]]; then
            echo "  PRESENT $TASK_NAME bytes=$(stat -c %s "$TASK_PATH")"
        else
            echo "  ABSENT $TASK_NAME"
        fi
    done
done
if [[ -s "$TASK_EVAL_LAUNCH_LOG" ]]; then
    echo '=== eval launcher tail ==='
    tail -100 "$TASK_EVAL_LAUNCH_LOG"
fi

echo '=== finetune artifact counts ==='
if [[ -d "$TASK_FINETUNE_OUTPUT" ]]; then
    echo "json_count=$(find "$TASK_FINETUNE_OUTPUT" -maxdepth 1 -type f -name '*.json' | wc -l)"
    echo "log_count=$(find "$TASK_FINETUNE_OUTPUT" -maxdepth 1 -type f -name '*.log' | wc -l)"
    echo "model_count=$(find "$TASK_FINETUNE_OUTPUT/models" -type f -name model_state.pt 2>/dev/null | wc -l)"
    test ! -s "$TASK_FINETUNE_OUTPUT/summary.md" || tail -120 "$TASK_FINETUNE_OUTPUT/summary.md"
else
    echo 'FINETUNE_OUTPUT_ABSENT'
fi

echo '=== burn progress tails ==='
for TASK_BURN_LOG in \
    /tmp/llm_pretrain_burn_eval_gpus_5_7.log \
    /tmp/llm_pretrain_burn_all_gpus.log; do
    if [[ -s "$TASK_BURN_LOG" ]]; then
        echo "burn_log=$TASK_BURN_LOG"
        tail -20 "$TASK_BURN_LOG"
    fi
done
echo 'TH2 FIVE-CHECKPOINT WORKFLOW READ-ONLY COMPLETION CHECK FINISHED'
