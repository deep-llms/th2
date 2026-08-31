#1 +45+a
#th2-readonly-check-gpu-usage-20260831-a01
set -euo pipefail

echo '=== identity ==='
date -u
hostname
git rev-parse HEAD

echo '=== static GPU inventory ==='
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader

echo '=== five utilization samples, two seconds apart ==='
for TASK_SAMPLE in 1 2 3 4 5; do
  echo "sample=$TASK_SAMPLE utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader
  if [[ "$TASK_SAMPLE" -lt 5 ]]; then
    sleep 2
  fi
done

echo '=== GPU compute applications ==='
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
  --format=csv,noheader

echo '=== compute-process ancestry ==='
mapfile -t TASK_GPU_PIDS < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader \
    | sed 's/^[[:space:]]*//;s/[[:space:]]*$//;/^$/d' | sort -nu
)
printf 'unique_gpu_pid_count=%s\n' "${#TASK_GPU_PIDS[@]}"
for TASK_PID in "${TASK_GPU_PIDS[@]}"; do
  case "$TASK_PID" in
    *[!0-9]*|'') echo "invalid GPU PID: $TASK_PID" >&2; exit 1 ;;
  esac
  ps -o pid=,ppid=,pgid=,stat=,etime=,cmd= -p "$TASK_PID"
done

echo '=== runner burn files, if present ==='
if [[ -f /tmp/llm_pretrain_burn.py ]]; then
  sha256sum /tmp/llm_pretrain_burn.py
fi
if [[ -f /tmp/llm_pretrain_burn_all_gpus.pid ]]; then
  printf 'launcher_pid='
  tr -d '[:space:]' </tmp/llm_pretrain_burn_all_gpus.pid
  printf '\n'
fi
if [[ -f /tmp/llm_pretrain_burn_all_gpus.log ]]; then
  tail -40 /tmp/llm_pretrain_burn_all_gpus.log
fi

echo 'TH2 READONLY GPU USAGE CHECK COMPLETE'
