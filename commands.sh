#1 +60+a
#th2-check-full-recent-evals-runtime-20260824
#!/usr/bin/env bash
set -euo pipefail

TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_INDEPENDENT_CKPT="$TASK_OUTPUT_BASE/lowrank_independent_output_r128/checkpoint-10000"
TASK_SHARED_LOCAL_CKPT="$TASK_OUTPUT_BASE/shared_local_tied_g16/checkpoint-10000"
TASK_LAUNCH_LOG="$TASK_OUTPUT_BASE/eval_parallel_independent_lr128_shared_local_g16_10k.log"

echo '=== live evaluation processes ==='
date -u
hostname
pgrep -af '[e]val_parallel.py|[e]val_checkpoint.py' || true

echo '=== live GPU state ==='
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader

echo '=== parallel launcher ==='
test -s "$TASK_LAUNCH_LOG"
tail -40 "$TASK_LAUNCH_LOG"

for checkpoint in "$TASK_INDEPENDENT_CKPT" "$TASK_SHARED_LOCAL_CKPT"; do
    echo "=== checkpoint: $checkpoint ==="
    test -s "$checkpoint/eval.log"
    ls -lh "$checkpoint/eval.log" "$checkpoint"/eval_*.json 2>/dev/null || true
    tail -60 "$checkpoint/eval.log"
done

if pgrep -f '[e]val_parallel.py|[e]val_checkpoint.py' >/dev/null; then
    echo 'TH2 FULL EVAL RUNTIME CHECK: BOTH EVALUATIONS STILL RUNNING'
else
    echo 'TH2 FULL EVAL RUNTIME CHECK: EVALUATION PROCESSES EXITED; INSPECT RESULTS ABOVE'
fi
