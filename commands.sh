#1 +60+a
#th2-relaunch-llm-pretrain-burn-20260822
set -euo pipefail

echo '=== th2 relaunch GPU burn ==='
date -u
hostname

TASK_BURN_SCRIPT="/tmp/llm_pretrain_burn.py"
test -s "$TASK_BURN_SCRIPT"

TASK_GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
echo "gpu_count=$TASK_GPU_COUNT"
test "$TASK_GPU_COUNT" -eq 8

echo '=== preflight: GPUs must be free ==='
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
TASK_EXISTING_GPU_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')"
if [ -n "$TASK_EXISTING_GPU_PIDS" ]; then
    echo 'REFUSE: GPU compute processes already exist'
    exit 1
fi

echo '=== launch one burn worker per GPU ==='
for TASK_GPU in 0 1 2 3 4 5 6 7; do
    TASK_LOG="/tmp/llm_pretrain_burn_gpu${TASK_GPU}.log"
    nohup env CUDA_VISIBLE_DEVICES="$TASK_GPU" /usr/bin/python3 -u "$TASK_BURN_SCRIPT" >"$TASK_LOG" 2>&1 &
    TASK_PID=$!
    echo "launched gpu=$TASK_GPU pid=$TASK_PID log=$TASK_LOG"
done

sleep 20

echo '=== post-launch verification ==='
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
TASK_BURN_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d' | sort -nu)"
TASK_BURN_COUNT="$(printf '%s\n' "$TASK_BURN_PIDS" | sed '/^[[:space:]]*$/d' | wc -l)"
echo "burn_process_count=$TASK_BURN_COUNT"
test "$TASK_BURN_COUNT" -eq 8

for TASK_PID in $TASK_BURN_PIDS; do
    TASK_CMDLINE="$(tr '\0' ' ' < "/proc/$TASK_PID/cmdline")"
    echo "gpu_pid=$TASK_PID cmdline=$TASK_CMDLINE"
    printf '%s\n' "$TASK_CMDLINE" | grep -Fq "$TASK_BURN_SCRIPT"
done

echo 'TH2 GPU BURN: RUNNING ON ALL 8 GPUS'
