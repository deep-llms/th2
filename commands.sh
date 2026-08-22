#1 +60+a
#th2-verify-relaunched-llm-pretrain-burn-20260822
set -euo pipefail

echo '=== th2 verify relaunched GPU burn ==='
date -u
hostname

TASK_BURN_SCRIPT="/tmp/llm_pretrain_burn.py"
test -s "$TASK_BURN_SCRIPT"

echo '=== GPU state ==='
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader

TASK_GPU_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d' | sort -nu)"
TASK_GPU_COUNT="$(printf '%s\n' "$TASK_GPU_PIDS" | sed '/^[[:space:]]*$/d' | wc -l)"
echo "gpu_compute_process_count=$TASK_GPU_COUNT"
test "$TASK_GPU_COUNT" -eq 8

for TASK_PID in $TASK_GPU_PIDS; do
    TASK_CMDLINE="$(tr '\0' ' ' < "/proc/$TASK_PID/cmdline")"
    echo "gpu_pid=$TASK_PID cmdline=$TASK_CMDLINE"
    printf '%s\n' "$TASK_CMDLINE" | grep -Fq "$TASK_BURN_SCRIPT"
done

echo 'TH2 GPU BURN VERIFIED: RUNNING ON ALL 8 GPUS'
