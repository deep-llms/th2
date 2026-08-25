#1 +120+a
#th2-eval-global-lr-tied-r128-and-dense-tied-b200-10k-20260825
#!/usr/bin/env bash
set -euo pipefail

TASK_PROJECT_DIR="$PWD"
TASK_EVAL_PYTHON=/mnt/local/conda-py311/envs/eval/bin/python3.11
TASK_BENCH_ROOT=/mnt/local/_data/@PROJECT@/benchmarks/hf
TASK_EVAL_DIR=/mnt/local/_data/@PROJECT@/data/Qwen_Qwen3-0.6B/eval
TASK_MODEL_DIR=/mnt/local/_models/@PROJECT@/Qwen3-0.6B
TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_GLOBAL_CKPT="$TASK_OUTPUT_BASE/global_lowrank_tied_r128_b200/checkpoint-10000"
TASK_DENSE_CKPT="$TASK_OUTPUT_BASE/dense_tied_baseline_b200/checkpoint-10000"
TASK_LAUNCH_LOG="$TASK_OUTPUT_BASE/eval_parallel_global_lr_tied_r128_dense_tied_b200_10k_20260825.log"

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

echo '=== host, environment, and idle-GPU preflight ==='
date -u
hostname
cd "$TASK_PROJECT_DIR"
test -x "$TASK_EVAL_PYTHON"
"$TASK_EVAL_PYTHON" -c 'import datasets, lm_eval, torch, transformers; print("eval_imports=OK", torch.__version__, transformers.__version__, datasets.__version__)'

mapfile -t TASK_GPU_NAMES < <(
    nvidia-smi --query-gpu=name --format=csv,noheader \
        | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
)
[[ "${#TASK_GPU_NAMES[@]}" -eq 8 ]] \
    || die "expected 8 GPUs, found ${#TASK_GPU_NAMES[@]}"
for index in "${!TASK_GPU_NAMES[@]}"; do
    [[ "${TASK_GPU_NAMES[$index]}" == *B200* ]] \
        || die "GPU $index is not a B200: ${TASK_GPU_NAMES[$index]}"
done
mapfile -t TASK_GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -nu
)
[[ "${#TASK_GPU_PIDS[@]}" -eq 0 ]] \
    || die "GPU compute processes exist before evaluation: ${TASK_GPU_PIDS[*]}"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
echo 'TH2 ALL 8 B200 GPUS FREE BEFORE TWO-MODEL EVAL'

echo '=== validate both exact checkpoint-10000 inputs and loader routing ==='
"$TASK_EVAL_PYTHON" - "$TASK_GLOBAL_CKPT" "$TASK_DENSE_CKPT" <<'PY'
import json
import os
import sys

from compositional.loading import is_compositional

global_ckpt, dense_ckpt = sys.argv[1:]
for checkpoint in (global_ckpt, dense_ckpt):
    assert os.path.isdir(checkpoint), checkpoint
    for filename in ("config.json", "model.safetensors", "trainer_state.json"):
        path = os.path.join(checkpoint, filename)
        assert os.path.isfile(path) and os.path.getsize(path) > 0, path
    with open(os.path.join(checkpoint, "trainer_state.json")) as handle:
        assert json.load(handle)["global_step"] == 10000, checkpoint

assert is_compositional(global_ckpt), global_ckpt
assert os.path.getsize(os.path.join(global_ckpt, "embedding.pt")) > 0
assert not os.path.exists(os.path.join(global_ckpt, "output_head.pt"))

assert not is_compositional(dense_ckpt), dense_ckpt
assert not os.path.exists(os.path.join(dense_ckpt, "embedding.pt"))
assert not os.path.exists(os.path.join(dense_ckpt, "output_head.pt"))
with open(os.path.join(dense_ckpt, "config.json")) as handle:
    dense_config = json.load(handle)
assert dense_config.get("tie_word_embeddings") is True, dense_config.get("tie_word_embeddings")
print("CHECKPOINT_ROUTING_OK global=compositional_lowrank_tied dense=native_dense_tied step=10000")
PY

test -s "$TASK_MODEL_DIR/config.json"
test -s "$TASK_MODEL_DIR/tokenizer.json"
for language in en vi zh ru de ar; do
    test -d "$TASK_EVAL_DIR/$language"
    test "$(find "$TASK_EVAL_DIR/$language" -type f -name '*.arrow' | wc -l)" -gt 0
done

echo '=== validate all offline benchmark repositories and 26 task mappings ==='
for relpath in \
    facebook/xnli facebook/belebele cambridgeltl/xcopa \
    juletxara/xstory_cloze google-research-datasets/paws-x \
    Rowan/hellaswag alexandrainst/m_hellaswag; do
    test -d "$TASK_BENCH_ROOT/$relpath" || die "missing offline dataset: $relpath"
    test "$(find "$TASK_BENCH_ROOT/$relpath" -type f | wc -l)" -gt 0
    echo "offline_dataset_ok=$relpath"
done
"$TASK_EVAL_PYTHON" - <<'PY'
import os

import lm_eval

from eval.benchmarks import TASK_CONFIGS, _DATASET_PATH_PATCHES, patch_lm_eval_dataset_paths

root = os.environ["LM_EVAL_DATASET_ROOT"]
patch_lm_eval_dataset_paths(root)
tasks_dir = os.path.join(os.path.dirname(lm_eval.__file__), "tasks")
for relative_file, repository, _aliases in _DATASET_PATH_PATCHES:
    expected = f"dataset_path: {os.path.join(root, repository)}"
    with open(os.path.join(tasks_dir, relative_file), encoding="utf-8") as handle:
        actual = [line.strip() for line in handle if line.startswith("dataset_path:")]
    assert actual == [expected], (relative_file, actual, expected)
tasks = [task for group in TASK_CONFIGS.values() for task in group]
assert len(tasks) == len(set(tasks)) == 26, tasks
print("OFFLINE_BENCHMARK_MAPPING_OK repositories=7 tasks=26")
PY

echo '=== refuse to overwrite any prior evaluation result ==='
for artifact in \
    "$TASK_GLOBAL_CKPT/eval.log" \
    "$TASK_GLOBAL_CKPT/eval_ppl.json" \
    "$TASK_GLOBAL_CKPT/eval_benchmarks.json" \
    "$TASK_DENSE_CKPT/eval.log" \
    "$TASK_DENSE_CKPT/eval_ppl.json" \
    "$TASK_DENSE_CKPT/eval_benchmarks.json" \
    "$TASK_LAUNCH_LOG"; do
    [[ ! -e "$artifact" ]] || die "refusing to overwrite evaluation artifact: $artifact"
done

echo '=== one eval_parallel.py run: two checkpoint-10000 models in parallel ==='
"$TASK_EVAL_PYTHON" -u eval/eval_parallel.py \
    --checkpoints "$TASK_GLOBAL_CKPT" "$TASK_DENSE_CKPT" \
    --eval-dir "$TASK_EVAL_DIR" \
    --tokenizer-name "$TASK_MODEL_DIR" \
    --bf16 \
    --num-gpus 8 \
    --log "$TASK_LAUNCH_LOG"

echo '=== validate complete PPL and benchmark outputs ==='
"$TASK_EVAL_PYTHON" - "$TASK_GLOBAL_CKPT" "$TASK_DENSE_CKPT" <<'PY'
import json
import math
import os
import sys

expected_languages = {"en", "vi", "zh", "ru", "de", "ar"}
for checkpoint in sys.argv[1:]:
    with open(os.path.join(checkpoint, "eval_ppl.json"), encoding="utf-8") as handle:
        perplexity = json.load(handle)
    with open(os.path.join(checkpoint, "eval_benchmarks.json"), encoding="utf-8") as handle:
        benchmarks = json.load(handle)
    assert set(perplexity) == expected_languages, (checkpoint, perplexity.keys())
    for language, metrics in perplexity.items():
        assert int(metrics["num_tokens"]) > 0, (checkpoint, language, metrics)
        assert math.isfinite(float(metrics["loss"])), (checkpoint, language, metrics)
        assert math.isfinite(float(metrics["perplexity"])), (checkpoint, language, metrics)
    assert len(benchmarks) == 26, (checkpoint, len(benchmarks))
    for task, metrics in benchmarks.items():
        accuracy = metrics.get("acc,none", metrics.get("acc"))
        assert accuracy is not None and math.isfinite(float(accuracy)), (
            checkpoint, task, metrics
        )
    print(f"EVAL_JSON_OK checkpoint={checkpoint} ppl_languages=6 benchmark_tasks=26")
PY

grep -F 'Loaded compositional model: arm=lowrank' "$TASK_GLOBAL_CKPT/eval.log"
if grep -Fq 'Loaded compositional model:' "$TASK_DENSE_CKPT/eval.log"; then
    die 'dense checkpoint was incorrectly routed through the compositional loader'
fi
if grep -HniE 'Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|FAILED \(code|Error:' \
    "$TASK_GLOBAL_CKPT/eval.log" "$TASK_DENSE_CKPT/eval.log" "$TASK_LAUNCH_LOG"; then
    die 'failure signature found in evaluation logs'
fi

mapfile -t TASK_FINAL_GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -nu
)
[[ "${#TASK_FINAL_GPU_PIDS[@]}" -eq 0 ]] \
    || die "GPU processes remain after evaluation: ${TASK_FINAL_GPU_PIDS[*]}"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
tail -100 "$TASK_LAUNCH_LOG"
echo 'TH2 GLOBAL LOWRANK TIED R128 AND DENSE TIED 10K FULL EVAL COMPLETE; GPUS FREE'
