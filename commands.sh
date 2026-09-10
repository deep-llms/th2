#1 +120+a
#th2-swt-readonly-next-capacity-current-progress-20260910-a04
set -euo pipefail

OUT=/mnt/local/_outputs/deep-llms_th2/swt/next_capacity_14arms_5k_s42_20260909_a01

echo '=== timestamp ==='
date -u +%Y-%m-%dT%H:%M:%SZ

echo '=== gpu summary ==='
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

echo '=== gpu compute processes ==='
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader

echo '=== pipeline markers ==='
grep -aE 'START_TRAINING_ARM|ARM_VERIFIED|ALL_TRAINING_COMPLETED_AND_VERIFIED|SWT_TRAINING_AND_BURN_HANDOFF_COMPLETE|SWT_PIPELINE_FAILED|Traceback|CUDA out of memory|OutOfMemoryError' "$OUT/pipeline.log" | tail -n 120 || true

echo '=== pipeline tail ==='
tail -c 120000 "$OUT/pipeline.log" | tr '\r' '\n' | tail -n 240

echo '=== output directories ==='
find "$OUT" -maxdepth 2 -type d -name 'checkpoint-*' -printf '%p\n' | sort -V | tail -n 80

echo 'SWT READONLY CURRENT PROGRESS COMPLETE'
