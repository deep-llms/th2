#1 +60+a
#th2-check-shared-local-10k-eval-completion-20260824
#!/usr/bin/env bash
set -euo pipefail

TASK_EVAL_PYTHON=/mnt/local/conda-py311/envs/eval/bin/python3.11
TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_CHECKPOINT="$TASK_OUTPUT_BASE/shared_local_tied_g16/checkpoint-10000"
TASK_LAUNCH_LOG="$TASK_OUTPUT_BASE/eval_parallel_shared_local_g16_10k_rerun_20260824.log"

echo '=== SharedLocal checkpoint-10000 evaluation status ==='
date -u
hostname

echo '=== evaluation processes ==='
pgrep -af '[e]val_parallel.py|[e]val_checkpoint.py' || true

echo '=== GPU state ==='
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader \
    || true

echo '=== launcher and checkpoint artifacts ==='
test -s "$TASK_LAUNCH_LOG"
tail -100 "$TASK_LAUNCH_LOG"
ls -lh "$TASK_CHECKPOINT/eval.log" "$TASK_CHECKPOINT/eval_ppl.json" \
    "$TASK_CHECKPOINT/eval_benchmarks.json" 2>/dev/null \
    || true
tail -100 "$TASK_CHECKPOINT/eval.log"

if pgrep -f '[e]val_parallel.py|[e]val_checkpoint.py' >/dev/null; then
    echo 'TH2 SHARED LOCAL 10K EVAL STATUS: STILL RUNNING'
    exit 0
fi

echo '=== validate completed evaluation ==='
"$TASK_EVAL_PYTHON" - "$TASK_CHECKPOINT" <<'PY'
import json
import math
import os
import sys

checkpoint = sys.argv[1]
with open(os.path.join(checkpoint, 'eval_ppl.json'), encoding='utf-8') as handle:
    perplexity = json.load(handle)
with open(os.path.join(checkpoint, 'eval_benchmarks.json'), encoding='utf-8') as handle:
    benchmarks = json.load(handle)
assert set(perplexity) == {'en', 'vi', 'zh', 'ru', 'de', 'ar'}
for language, metrics in perplexity.items():
    assert int(metrics['num_tokens']) > 0, (language, metrics)
    assert math.isfinite(float(metrics['loss'])), (language, metrics)
    assert math.isfinite(float(metrics['perplexity'])), (language, metrics)
assert len(benchmarks) == 26, len(benchmarks)
for task, metrics in benchmarks.items():
    accuracy = metrics.get('acc,none', metrics.get('acc'))
    assert accuracy is not None and math.isfinite(float(accuracy)), (task, metrics)
print('SHARED_LOCAL_10K_EVAL_JSON_OK ppl_languages=6 benchmark_tasks=26')
PY

grep -F 'Loaded compositional model: arm=shared_local' "$TASK_CHECKPOINT/eval.log"
grep -F 'All 1 evaluations done' "$TASK_LAUNCH_LOG"
if grep -HniE 'Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|FAILED \(code|Error:' \
    "$TASK_CHECKPOINT/eval.log" "$TASK_LAUNCH_LOG"; then
    echo 'ERROR: failure signature found in SharedLocal evaluation logs'
    exit 1
fi
mapfile -t TASK_GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -nu
)
[[ "${#TASK_GPU_PIDS[@]}" -eq 0 ]] || {
    echo "ERROR: GPU processes remain after evaluation: ${TASK_GPU_PIDS[*]}"
    exit 1
}
echo 'TH2 SHARED LOCAL 10K EVAL COMPLETE AND VERIFIED; ALL GPUS FREE'
