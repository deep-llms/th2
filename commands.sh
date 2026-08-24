#1 +60+a
#th2-check-full-recent-evals-completion-20260824
#!/usr/bin/env bash
set -euo pipefail

TASK_EVAL_PYTHON=/mnt/local/conda-py311/envs/eval/bin/python3.11
TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_INDEPENDENT_CKPT="$TASK_OUTPUT_BASE/lowrank_independent_output_r128/checkpoint-10000"
TASK_SHARED_LOCAL_CKPT="$TASK_OUTPUT_BASE/shared_local_tied_g16/checkpoint-10000"
TASK_LAUNCH_LOG="$TASK_OUTPUT_BASE/eval_parallel_independent_lr128_shared_local_g16_10k.log"

echo '=== evaluation processes ==='
date -u
hostname
pgrep -af '[e]val_parallel.py|[e]val_checkpoint.py' || true

echo '=== GPU state ==='
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader

echo '=== launcher tail ==='
test -s "$TASK_LAUNCH_LOG"
tail -80 "$TASK_LAUNCH_LOG"

echo '=== artifacts and per-model log tails ==='
for checkpoint in "$TASK_INDEPENDENT_CKPT" "$TASK_SHARED_LOCAL_CKPT"; do
    echo "--- $checkpoint ---"
    ls -lh "$checkpoint/eval.log" "$checkpoint"/eval_*.json 2>/dev/null || true
    tail -80 "$checkpoint/eval.log"
done

if pgrep -f '[e]val_parallel.py|[e]val_checkpoint.py' >/dev/null; then
    echo 'TH2 FULL EVAL STATUS: STILL RUNNING'
    exit 0
fi

echo '=== validate completed JSON results ==='
"$TASK_EVAL_PYTHON" - "$TASK_INDEPENDENT_CKPT" "$TASK_SHARED_LOCAL_CKPT" <<'PY'
import json
import math
import os
import sys

expected_languages = {'en', 'vi', 'zh', 'ru', 'de', 'ar'}
for checkpoint in sys.argv[1:]:
    with open(os.path.join(checkpoint, 'eval_ppl.json'), encoding='utf-8') as handle:
        perplexity = json.load(handle)
    with open(os.path.join(checkpoint, 'eval_benchmarks.json'), encoding='utf-8') as handle:
        benchmarks = json.load(handle)
    assert set(perplexity) == expected_languages, (checkpoint, perplexity.keys())
    for language, metrics in perplexity.items():
        assert int(metrics['num_tokens']) > 0, (checkpoint, language, metrics)
        assert math.isfinite(float(metrics['loss'])), (checkpoint, language, metrics)
        assert math.isfinite(float(metrics['perplexity'])), (checkpoint, language, metrics)
    assert len(benchmarks) == 26, (checkpoint, len(benchmarks))
    for task, metrics in benchmarks.items():
        accuracy = metrics.get('acc,none', metrics.get('acc'))
        assert accuracy is not None and math.isfinite(float(accuracy)), (
            checkpoint, task, metrics
        )
    print(f"COMPLETE_JSON_OK checkpoint={checkpoint} ppl_languages=6 benchmark_tasks=26")
PY

grep -F 'Loaded compositional model: arm=lowrank' "$TASK_INDEPENDENT_CKPT/eval.log"
grep -F 'Loaded compositional model: arm=shared_local' "$TASK_SHARED_LOCAL_CKPT/eval.log"
if grep -HniE 'Traceback \(most recent call last\)|CUDA out of memory|FAILED \(code|Error:' \
    "$TASK_INDEPENDENT_CKPT/eval.log" "$TASK_SHARED_LOCAL_CKPT/eval.log"; then
    echo 'ERROR: failure signature found in evaluation logs'
    exit 1
fi

if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; then
    echo 'ERROR: GPU compute processes remain after completed evaluation'
    exit 1
fi
echo 'TH2 FULL EVAL STATUS: COMPLETE AND VERIFIED; ALL GPUS FREE'
