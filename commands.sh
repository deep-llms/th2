#1 +60+a
#th2-stop-llm-pretrain-burn-and-verify-free-20260822
set -euo pipefail

echo '=== th2 validated GPU-burn cleanup ==='
date -u
hostname

echo '=== before cleanup ==='
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
TASK_GPU_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d' | sort -nu)"

if [ -z "$TASK_GPU_PIDS" ]; then
    echo 'TH2 GPU CLEANUP: ALREADY FREE'
    exit 0
fi

echo '=== validate every GPU process before killing anything ==='
for TASK_PID in $TASK_GPU_PIDS; do
    [[ "$TASK_PID" =~ ^[0-9]+$ ]] || {
        echo "ERROR: invalid GPU PID: $TASK_PID"
        exit 1
    }
    TASK_CMDLINE="$(tr '\0' ' ' < "/proc/$TASK_PID/cmdline" 2>/dev/null || true)"
    ps -o pid,ppid,user,stat,etime,cmd -p "$TASK_PID" || true
    echo "gpu_pid=$TASK_PID cmdline=$TASK_CMDLINE"
    if ! printf '%s\n' "$TASK_CMDLINE" | grep -Fq '/tmp/llm_pretrain_burn.py'; then
        echo "REFUSE: GPU PID $TASK_PID is not /tmp/llm_pretrain_burn.py"
        exit 1
    fi
done

TASK_CURRENT_GPU_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d' | sort -nu)"
if [ "$TASK_CURRENT_GPU_PIDS" != "$TASK_GPU_PIDS" ]; then
    echo 'REFUSE: GPU process set changed during validation'
    exit 1
fi

echo '=== terminate validated burn processes ==='
for TASK_PID in $TASK_GPU_PIDS; do
    kill -TERM "$TASK_PID"
done
sleep 5

TASK_REMAINING_GPU_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d' | sort -nu)"
for TASK_PID in $TASK_REMAINING_GPU_PIDS; do
    if printf '%s\n' "$TASK_GPU_PIDS" | grep -qx "$TASK_PID"; then
        echo "force killing validated burn PID $TASK_PID"
        kill -KILL "$TASK_PID"
    else
        echo "REFUSE: new GPU PID appeared after termination: $TASK_PID"
        exit 1
    fi
done
sleep 5

echo '=== after cleanup ==='
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
TASK_REMAINING_GPU_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')"
test -z "$TASK_REMAINING_GPU_PIDS"

echo 'TH2 GPU CLEANUP: ALL 8 GPUS FREE'
