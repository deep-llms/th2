#1 +60+a
#th2-check-recent-finetune-runtime-20260824
#!/usr/bin/env bash
set -euo pipefail

TASK_OUTPUT=/mnt/local/_outputs/@PROJECT@/finetune_independent_lr128_shared_local_g16

echo '=== finetune processes ==='
date -u
hostname
pgrep -af '[f]inetune/run_all.py|[f]inetune/train.py' || true

echo '=== GPU state ==='
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader

echo '=== artifacts so far ==='
test -d "$TASK_OUTPUT"
echo "logs=$(find "$TASK_OUTPUT" -maxdepth 1 -type f -name '*.log' | wc -l)"
echo "jsons=$(find "$TASK_OUTPUT" -maxdepth 1 -type f -name '*.json' | wc -l)"
echo "models=$(find "$TASK_OUTPUT/models" -type f -name 'model_state.pt' 2>/dev/null | wc -l)"
du -sh "$TASK_OUTPUT"

echo '=== failure scan ==='
if grep -HniE 'traceback|out of memory|nan|eval failed:|FAILED \(code' "$TASK_OUTPUT"/*.log; then
    echo 'ERROR: failure signature found in finetune logs'
    exit 1
else
    echo 'no failure signatures'
fi

echo '=== active log tails ==='
for log_path in "$TASK_OUTPUT"/*.log; do
    echo "--- $log_path"
    tail -20 "$log_path"
done

if pgrep -f '[f]inetune/run_all.py|[f]inetune/train.py' >/dev/null; then
    echo 'TH2 FINETUNE RUNTIME CHECK: RUNNING'
else
    echo 'TH2 FINETUNE RUNTIME CHECK: PROCESSES EXITED; INSPECT ARTIFACTS ABOVE'
fi
