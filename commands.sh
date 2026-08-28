#1 +30+a
#th2-check-gpu-usage-and-burns-during-culturax-sampling-20260828-a01
set -euo pipefail

echo '=== timestamp and GPU summary ==='
date -u
hostname
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader

echo '=== GPU compute applications ==='
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits || true

echo '=== unique GPU PID command lines ==='
mapfile -t TASK_GPU_PIDS < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -u
)
echo "unique_gpu_pid_count=${#TASK_GPU_PIDS[@]}"
for TASK_PID in "${TASK_GPU_PIDS[@]}"; do
  if [ -r "/proc/$TASK_PID/cmdline" ]; then
    TASK_CMD="$(tr '\0' ' ' < "/proc/$TASK_PID/cmdline")"
    TASK_COMM="$(cat "/proc/$TASK_PID/comm" 2>/dev/null || true)"
    echo "gpu_pid=$TASK_PID comm=$TASK_COMM cmd=$TASK_CMD"
  else
    echo "gpu_pid=$TASK_PID cmdline=UNAVAILABLE"
  fi
done

echo '=== active sampling process ==='
ps -eo pid,ppid,etime,pcpu,pmem,args \
  | awk 'NR == 1 || /[p]repare_data\.py.*sample/'

echo 'TH2 GPU USAGE INSPECTION OK'
