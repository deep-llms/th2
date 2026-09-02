#1 +45+a
#th2-readonly-ranklift-smoke-progress-20260902-a01
set -euo pipefail

TASK_OUTPUT=/mnt/local/_outputs/@PROJECT@/final_interfaces_smoke_20260902

echo '=== gpu state ==='
date -u
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits || true

echo '=== RankLift 50-step live log ==='
if [[ -s "$TASK_OUTPUT/ranklift_qwen_50step.log" ]]; then
    tr '\r' '\n' < "$TASK_OUTPUT/ranklift_qwen_50step.log" | tail -n 100
else
    echo 'ranklift smoke log not present yet'
fi

echo '=== smoke files ==='
find "$TASK_OUTPUT" -maxdepth 3 -type f -printf '%s %p\n' 2>/dev/null | sort || true
echo 'TH2 READONLY RANKLIFT SMOKE PROGRESS COMPLETE'
