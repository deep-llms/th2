#1 +60+a
#th2-swt-readonly-gpu-memory-20260908-1441-a01
set -euo pipefail
date -u
hostname
test "$(hostname)" = thiennh-p6-oish-worker-0
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv
echo SWT_READONLY_GPU_MEMORY_SNAPSHOT_COMPLETE
