#1 +30+a
#th2-8mgy-readonly-gpu-usage-20260911-a01
set -euo pipefail
date -u
hostname
test "$(hostname)" = thiennh-p6-8mgy-worker-0
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
sleep 5
date -u
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
echo GPU_USAGE_CHECK_COMPLETE
