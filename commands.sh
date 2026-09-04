#1 +120+a
#th2-readonly-verify-post-cancel-full-burn-20260904-a01
set -u

TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_FINETUNE_OUTPUT="$TASK_OUTPUT_BASE/finetune_ranklift_hashedv2_btmos_steps_6500_10000_20260904_a01"
TASK_BURN_LOG=/tmp/llm_pretrain_burn_all_gpus.log
TASK_BURN_PID_FILE=/tmp/llm_pretrain_burn_launcher.pid

echo '=== post-cancellation process state ==='
date -u
pgrep -af '[f]inetune/run_all.py' || true
pgrep -af '[f]inetune/train.py' || true
pgrep -af '[r]un_five_checkpoint_eval_finetune_burn.sh' || true
pgrep -af '[c]ancel-five-checkpoint-finetune-and-restore-burn' || true
echo "json_count=$(find "$TASK_FINETUNE_OUTPUT" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l)"
echo "model_count=$(find "$TASK_FINETUNE_OUTPUT/models" -type f -name model_state.pt 2>/dev/null | wc -l)"

echo '=== burn launcher and log ==='
if [[ -s "$TASK_BURN_PID_FILE" ]]; then
    TASK_LAUNCHER="$(tr -d '[:space:]' < "$TASK_BURN_PID_FILE")"
    echo "launcher=$TASK_LAUNCHER"
    if [[ "$TASK_LAUNCHER" =~ ^[0-9]+$ && "$TASK_LAUNCHER" -ne 1 ]]; then
        ps -o pid=,ppid=,etimes=,stat=,args= -p "$TASK_LAUNCHER" || true
    fi
else
    echo 'BURN_PID_FILE_ABSENT'
fi
if [[ -s "$TASK_BURN_LOG" ]]; then
    echo "ready_count=$(grep -Fc 'gpu_burn_ready' "$TASK_BURN_LOG" || true)"
    echo "world8_count=$(grep -Fc 'world_size=8' "$TASK_BURN_LOG" || true)"
    echo "progress_count=$(grep -Fc 'gpu_burn_progress' "$TASK_BURN_LOG" || true)"
    tail -40 "$TASK_BURN_LOG"
else
    echo 'BURN_LOG_ABSENT_OR_EMPTY'
fi

echo '=== GPU state ==='
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits || true
echo 'TH2 POST-CANCEL BURN READ-ONLY CHECK FINISHED'
