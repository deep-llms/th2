#1 +120+a
#th2-stop-unintended-resume-run-shared-local-10k-eval-20260824
#!/usr/bin/env bash
set -euo pipefail

TASK_EVAL_PYTHON=/mnt/local/conda-py311/envs/eval/bin/python3.11
TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_TRAIN_OUTPUT="$TASK_OUTPUT_BASE/shared_local_tied_g16"
TASK_CHECKPOINT="$TASK_OUTPUT_BASE/shared_local_tied_g16/checkpoint-10000"
TASK_EVAL_DIR=/mnt/local/_data/@PROJECT@/data/Qwen_Qwen3-0.6B/eval
TASK_MODEL_DIR=/mnt/local/_models/@PROJECT@/Qwen3-0.6B
TASK_BENCH_ROOT=/mnt/local/_data/@PROJECT@/benchmarks/hf
TASK_LAUNCH_LOG="$TASK_OUTPUT_BASE/eval_parallel_shared_local_g16_10k_rerun_20260824.log"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export LM_EVAL_DATASET_ROOT="$TASK_BENCH_ROOT"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

die() {
    echo "ERROR: $*"
    exit 1
}

list_target_train_pids() {
    ps -eo pid=,args= \
        | grep '[t]rain_compositional.py' \
        | grep -F -- "--output_dir $TASK_TRAIN_OUTPUT" \
        | grep -F -- '--run_name shared-local-tied-g16-qwen3-0.6b' \
        | awk '{print $1}' \
        || true
}

echo '=== stop only the unintended SharedLocal training continuation ==='
date -u
hostname
mapfile -t TASK_TRAIN_PIDS < <(list_target_train_pids)
if [[ "${#TASK_TRAIN_PIDS[@]}" -gt 0 ]]; then
    printf 'TERM training PID %s\n' "${TASK_TRAIN_PIDS[@]}"
    kill -TERM "${TASK_TRAIN_PIDS[@]}" || true
    for _ in $(seq 1 30); do
        mapfile -t TASK_REMAINING_PIDS < <(list_target_train_pids)
        [[ "${#TASK_REMAINING_PIDS[@]}" -gt 0 ]] || break
        sleep 1
    done
fi
mapfile -t TASK_TRAIN_PIDS < <(list_target_train_pids)
if [[ "${#TASK_TRAIN_PIDS[@]}" -gt 0 ]]; then
    printf 'KILL remaining training PID %s\n' "${TASK_TRAIN_PIDS[@]}"
    kill -KILL "${TASK_TRAIN_PIDS[@]}" || true
    sleep 5
fi
mapfile -t TASK_TRAIN_PIDS < <(list_target_train_pids)
[[ "${#TASK_TRAIN_PIDS[@]}" -eq 0 ]] \
    || die "target SharedLocal training processes remain: ${TASK_TRAIN_PIDS[*]}"
echo 'UNINTENDED SHAREDLOCAL CONTINUATION STOPPED'

echo '=== verify exact checkpoint-10000 and free GPUs ==='
test -x "$TASK_EVAL_PYTHON"
test -s "$TASK_CHECKPOINT/config.json"
test -s "$TASK_CHECKPOINT/model.safetensors"
test -s "$TASK_CHECKPOINT/embedding.pt"
test -s "$TASK_CHECKPOINT/trainer_state.json"
[[ ! -e "$TASK_CHECKPOINT/output_head.pt" ]] \
    || die 'SharedLocal tied checkpoint unexpectedly has output_head.pt'
for language in en vi zh ru de ar; do
    test -d "$TASK_EVAL_DIR/$language"
    test "$(find "$TASK_EVAL_DIR/$language" -type f -name '*.arrow' | wc -l)" -gt 0
done
for relpath in \
    facebook/xnli facebook/belebele cambridgeltl/xcopa \
    juletxara/xstory_cloze google-research-datasets/paws-x \
    Rowan/hellaswag alexandrainst/m_hellaswag; do
    test -d "$TASK_BENCH_ROOT/$relpath" || die "missing offline dataset: $relpath"
done

mapfile -t TASK_GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -nu
)
[[ "${#TASK_GPU_PIDS[@]}" -eq 0 ]] \
    || die "GPU processes remain before evaluation: ${TASK_GPU_PIDS[*]}"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
echo 'ALL 8 B200 GPUS FREE BEFORE EVAL'

echo '=== patch lm-eval to verified offline snapshots ==='
"$TASK_EVAL_PYTHON" - <<'PY'
import os
from eval.benchmarks import patch_lm_eval_dataset_paths

patch_lm_eval_dataset_paths(os.environ['LM_EVAL_DATASET_ROOT'])
print('OFFLINE_LM_EVAL_PATHS_OK')
PY

echo '=== run eval_parallel.py for SharedLocal tied G16 checkpoint-10000 ==='
"$TASK_EVAL_PYTHON" -u eval/eval_parallel.py \
    --checkpoints "$TASK_CHECKPOINT" \
    --eval-dir "$TASK_EVAL_DIR" \
    --tokenizer-name "$TASK_MODEL_DIR" \
    --bf16 \
    --num-gpus 8 \
    --log "$TASK_LAUNCH_LOG"

echo '=== validate complete SharedLocal evaluation ==='
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
print('SHARED_LOCAL_10K_EVAL_OK ppl_languages=6 benchmark_tasks=26')
PY
grep -F 'Loaded compositional model: arm=shared_local' "$TASK_CHECKPOINT/eval.log"
if grep -HniE 'Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|FAILED \(code|Error:' \
    "$TASK_CHECKPOINT/eval.log" "$TASK_LAUNCH_LOG"; then
    die 'failure signature found in SharedLocal evaluation logs'
fi
mapfile -t TASK_FINAL_GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -nu
)
[[ "${#TASK_FINAL_GPU_PIDS[@]}" -eq 0 ]] \
    || die "GPU processes remain after evaluation: ${TASK_FINAL_GPU_PIDS[*]}"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
echo 'TH2 SHARED LOCAL TIED G16 CHECKPOINT-10000 EVAL COMPLETE AND VERIFIED; ALL GPUS FREE'
