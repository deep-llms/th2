#1 +120+a
#th2-check-gpus-before-new-10k-screens
set -euo pipefail

echo '=== th2 GPU availability check ==='
date -u
hostname

echo '=== GPU inventory and utilization ==='
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
TASK_GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
echo "gpu_count=$TASK_GPU_COUNT"
test "$TASK_GPU_COUNT" -eq 8

echo '=== active GPU compute processes ==='
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
TASK_GPU_PROCESSES="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')"

echo '=== training-related host processes ==='
pgrep -af '[r]un_experiments.py|[t]rain_compositional.py|[t]rain_original_ant.py|[a]ccelerate launch|[f]inetune/' || echo 'none'

if [ -n "$TASK_GPU_PROCESSES" ]; then
    echo 'TH2 GPU CHECK: BUSY'
    exit 1
fi

echo 'TH2 GPU CHECK: ALL 8 GPUS FREE'
