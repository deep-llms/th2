#1 +60+a
#th2-diagnose-llm-pretrain-burn-exit-20260822
set -euo pipefail

echo '=== th2 diagnose GPU burn exit ==='
date -u
hostname

TASK_BURN_SCRIPT="/tmp/llm_pretrain_burn.py"
test -s "$TASK_BURN_SCRIPT"

echo '=== burn script metadata ==='
ls -l "$TASK_BURN_SCRIPT"
sha256sum "$TASK_BURN_SCRIPT"

echo '=== worker logs ==='
for TASK_GPU in 0 1 2 3 4 5 6 7; do
    TASK_LOG="/tmp/llm_pretrain_burn_gpu${TASK_GPU}.log"
    echo "--- gpu=$TASK_GPU log=$TASK_LOG ---"
    if [ -f "$TASK_LOG" ]; then
        tail -n 40 "$TASK_LOG"
    else
        echo 'MISSING LOG'
    fi
done

echo '=== current GPU state ==='
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader

echo 'TH2 GPU BURN DIAGNOSTIC COMPLETE'
