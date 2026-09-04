#1 +240+a
#th2-readonly-inspect-live-xnli-duration-20260904-a01
set -u

TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_FINETUNE_OUTPUT="$TASK_OUTPUT_BASE/finetune_ranklift_hashedv2_btmos_steps_6500_10000_20260904_a01"
TASK_COMPLETION_MARKER="$TASK_OUTPUT_BASE/eval_finetune_ranklift_hashedv2_btmos_steps_6500_10000_20260904_a01.complete"

echo '=== time and overall state ==='
date -u
if [[ -s "$TASK_COMPLETION_MARKER" ]]; then
    echo 'COMPLETION_MARKER_PRESENT'
    cat "$TASK_COMPLETION_MARKER"
else
    echo 'COMPLETION_MARKER_ABSENT'
fi
echo "json_count=$(find "$TASK_FINETUNE_OUTPUT" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l)"
echo "log_count=$(find "$TASK_FINETUNE_OUTPUT" -maxdepth 1 -type f -name '*.log' 2>/dev/null | wc -l)"
echo "model_count=$(find "$TASK_FINETUNE_OUTPUT/models" -type f -name model_state.pt 2>/dev/null | wc -l)"

echo '=== live finetune processes with elapsed seconds ==='
ps -eo pid=,lstart=,etimes=,args= | grep '[f]inetune/train.py' || true

echo '=== representative Hashed-V2 XNLI job ==='
TASK_HASHED_LOG="$TASK_FINETUNE_OUTPUT/xnli_hashedv2_h6144_s10000_seed456.log"
TASK_HASHED_JSON="$TASK_FINETUNE_OUTPUT/xnli_hashedv2_h6144_s10000_seed456.json"
if [[ -e "$TASK_HASHED_LOG" ]]; then
    stat -c 'log=%n bytes=%s modified=%y' "$TASK_HASHED_LOG"
    tail -100 "$TASK_HASHED_LOG"
else
    echo "MISSING_LOG $TASK_HASHED_LOG"
fi
if [[ -s "$TASK_HASHED_JSON" ]]; then
    echo 'HASHED_JOB_COMPLETE'
    cat "$TASK_HASHED_JSON"
else
    echo 'HASHED_JOB_NOT_COMPLETE'
fi

echo '=== representative BT-MoS XNLI job ==='
TASK_BTMOS_LOG="$TASK_FINETUNE_OUTPUT/xnli_btmos_k3_c256_lb_s6500_seed456.log"
TASK_BTMOS_JSON="$TASK_FINETUNE_OUTPUT/xnli_btmos_k3_c256_lb_s6500_seed456.json"
if [[ -e "$TASK_BTMOS_LOG" ]]; then
    stat -c 'log=%n bytes=%s modified=%y' "$TASK_BTMOS_LOG"
    tail -100 "$TASK_BTMOS_LOG"
else
    echo "MISSING_LOG $TASK_BTMOS_LOG"
fi
if [[ -s "$TASK_BTMOS_JSON" ]]; then
    echo 'BTMOS_JOB_COMPLETE'
    cat "$TASK_BTMOS_JSON"
else
    echo 'BTMOS_JOB_NOT_COMPLETE'
fi

echo '=== GPU state ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits || true

echo 'TH2 LIVE XNLI DURATION INSPECTION FINISHED'
