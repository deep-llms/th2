#1 +30+a
#th2-q359-ce7454-interim-watch-health-20261016-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
export CUDA_VISIBLE_DEVICES=''
TASK_ROOT=/mnt/local/_outputs/deep-llms_th2/q359-ce7454-interim-watch-20261016-a01
date -u
cat "$TASK_ROOT/installed.json"
cat "$TASK_ROOT/watch/latest_status.json"
cat "$TASK_ROOT/watch.log"
ps -p 1271652 -o pid,lstart,stat,args
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory --format=csv
