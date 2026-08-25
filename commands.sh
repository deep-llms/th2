#1 +60+a
#th2-check-matched-tied-eval-to-finetune-handoff-20260825
#!/usr/bin/env bash
set -euo pipefail

TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_GLOBAL_CKPT="$TASK_OUTPUT_BASE/global_lowrank_tied_r128_b200/checkpoint-10000"
TASK_DENSE_CKPT="$TASK_OUTPUT_BASE/dense_tied_baseline_b200/checkpoint-10000"
TASK_EVAL_LAUNCH_LOG="$TASK_OUTPUT_BASE/eval_parallel_global_lr_tied_r128_dense_tied_b200_10k_20260825.log"
TASK_FINETUNE_OUTPUT="$TASK_OUTPUT_BASE/finetune_global_lr_tied_r128_dense_tied_b200_10k_20260825"

echo '=== exact eval and finetune processes (read only) ==='
date -u
hostname
ps -eo pid=,etime=,args= | grep -E '[e]val/eval_parallel.py|[e]val/eval_checkpoint.py|[f]inetune/run_all.py|[f]inetune/train.py' || true

echo '=== GPU state (read only) ==='
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true

echo '=== evaluation launcher ==='
if [[ -s "$TASK_EVAL_LAUNCH_LOG" ]]; then
    tail -80 "$TASK_EVAL_LAUNCH_LOG"
else
    echo "missing_or_empty=$TASK_EVAL_LAUNCH_LOG"
fi

echo '=== evaluation artifacts and tails ==='
for checkpoint in "$TASK_GLOBAL_CKPT" "$TASK_DENSE_CKPT"; do
    echo "--- $checkpoint ---"
    ls -lh "$checkpoint/eval.log" "$checkpoint"/eval_*.json 2>/dev/null || true
    tail -40 "$checkpoint/eval.log" 2>/dev/null || true
done

echo '=== finetune handoff artifacts ==='
if [[ -d "$TASK_FINETUNE_OUTPUT" ]]; then
    find "$TASK_FINETUNE_OUTPUT" -maxdepth 1 -type f \
        \( -name '*.json' -o -name '*.log' -o -name 'summary.md' \) \
        -printf '%f %s bytes\n' | sort
else
    echo "finetune_output_absent=$TASK_FINETUNE_OUTPUT"
fi
echo 'TH2 EVAL TO FINETUNE HANDOFF STATUS CHECK FINISHED'
