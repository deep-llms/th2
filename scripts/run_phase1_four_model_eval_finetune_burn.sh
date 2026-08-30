#!/usr/bin/env bash
# Evaluate and finetune the two Nested-Ladder Phase-1 arms and both fresh-B200
# dense controls, then supervise one GPU-burn worker per B200.

set -euo pipefail

die() {
    echo "ERROR: $*" >&2
    exit 1
}

gpu_pids() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -nu
}

validate_b200_node() {
    local index
    mapfile -t TASK_GPU_NAMES < <(
        nvidia-smi --query-gpu=name --format=csv,noheader \
            | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
    )
    [[ "${#TASK_GPU_NAMES[@]}" -eq 8 ]] \
        || die "expected 8 GPUs, found ${#TASK_GPU_NAMES[@]}"
    for index in "${!TASK_GPU_NAMES[@]}"; do
        [[ "${TASK_GPU_NAMES[$index]}" == *B200* ]] \
            || die "GPU $index is not B200: ${TASK_GPU_NAMES[$index]}"
    done
}

require_free_gpus() {
    local stage="$1"
    mapfile -t TASK_STAGE_GPU_PIDS < <(gpu_pids)
    nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,power.draw \
        --format=csv,noheader
    [[ "${#TASK_STAGE_GPU_PIDS[@]}" -eq 0 ]] \
        || die "GPU processes remain $stage: ${TASK_STAGE_GPU_PIDS[*]}"
    echo "ALL 8 B200 GPUS FREE $stage"
}

scan_fatal_logs() {
    local label="$1"
    shift
    local pattern
    pattern='Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|ChildFailedError|ProcessExitedException|FAILED \(code|eval failed:|NCCL.*(unhandled|system error|remote process exited|watchdog|timeout)|Segmentation fault|Bus error'
    if grep -HniE "$pattern" "$@"; then
        die "fatal signature found in $label logs"
    fi
}

install_accelerate_config() {
    mkdir -p "$(dirname "$TASK_ACCELERATE_TARGET")"
    cp "$TASK_ACCELERATE_SOURCE" "$TASK_ACCELERATE_TARGET"
    cmp "$TASK_ACCELERATE_SOURCE" "$TASK_ACCELERATE_TARGET"
    grep -Fxq 'distributed_type: MULTI_GPU' "$TASK_ACCELERATE_TARGET"
    grep -Fxq 'mixed_precision: bf16' "$TASK_ACCELERATE_TARGET"
    grep -Fxq 'num_processes: 8' "$TASK_ACCELERATE_TARGET"
    echo "ACCELERATE_CONFIG_OK target=$TASK_ACCELERATE_TARGET"
}

TASK_PROJECT_DIR="${SPARSE_EMB_PROJECT_DIR:?SPARSE_EMB_PROJECT_DIR is required}"
TASK_OUTPUT_BASE="${SPARSE_EMB_OUTPUT_BASE:?SPARSE_EMB_OUTPUT_BASE is required}"
TASK_MODEL_DIR="${SPARSE_EMB_MODEL_DIR:?SPARSE_EMB_MODEL_DIR is required}"
TASK_EVAL_DIR="${SPARSE_EMB_EVAL_DIR:?SPARSE_EMB_EVAL_DIR is required}"
TASK_BENCH_ROOT="${SPARSE_EMB_BENCH_ROOT:?SPARSE_EMB_BENCH_ROOT is required}"
TASK_EVAL_PYTHON="${SPARSE_EMB_EVAL_PYTHON:?SPARSE_EMB_EVAL_PYTHON is required}"
TASK_CONDA="${SPARSE_EMB_CONDA:?SPARSE_EMB_CONDA is required}"

TASK_NESTED_CKPT="$TASK_OUTPUT_BASE/nested_ladder_tied_t4/checkpoint-10000"
TASK_GROUPREDUCE_CKPT="$TASK_OUTPUT_BASE/groupreduce_matched_nested_tied_t4/checkpoint-10000"
TASK_DENSE_CKPT="$TASK_OUTPUT_BASE/dense_tied_baseline_b200/checkpoint-10000"
TASK_DENSE_DDP_DEFAULT_CKPT="$TASK_OUTPUT_BASE/dense_tied_baseline_b200_ddp_default/checkpoint-10000"
TASK_CHECKPOINTS=(
    "$TASK_NESTED_CKPT"
    "$TASK_GROUPREDUCE_CKPT"
    "$TASK_DENSE_CKPT"
    "$TASK_DENSE_DDP_DEFAULT_CKPT"
)
TASK_EVAL_LAUNCH_LOG="$TASK_OUTPUT_BASE/eval_parallel_nested_groupreduce_two_dense_10k_20260830.log"
TASK_FINETUNE_OUTPUT="$TASK_OUTPUT_BASE/finetune_nested_groupreduce_two_dense_10k_20260830"
TASK_ACCELERATE_SOURCE="$TASK_PROJECT_DIR/resources/accelerate_config.yaml"
TASK_ACCELERATE_TARGET=/mnt/local/.cache/huggingface/accelerate/default_config.yaml
TASK_BURN_SCRIPT="$TASK_PROJECT_DIR/scripts/gpu_burn.py"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export LM_EVAL_DATASET_ROOT="$TASK_BENCH_ROOT"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo '=== four-model checkpoint-10k eval -> finetune -> burn workflow ==='
date -u
hostname
cd "$TASK_PROJECT_DIR"
test -x "$TASK_CONDA"
test -x "$TASK_EVAL_PYTHON"
test -s "$TASK_BURN_SCRIPT"
test -s "$TASK_ACCELERATE_SOURCE"
validate_b200_node

echo '=== activate and validate the eval environment ==='
eval "$("$TASK_CONDA" shell.bash hook)"
conda activate eval
[[ "${CONDA_DEFAULT_ENV:-}" == eval ]] || die 'failed to activate eval environment'
[[ "$(command -v python3.11)" == "$TASK_EVAL_PYTHON" ]] \
    || die "wrong Python after activation: $(command -v python3.11)"
"$TASK_EVAL_PYTHON" - <<'PY'
import datasets
import lm_eval
import torch
import transformers

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 8
print(
    "EVAL_ENV_OK",
    f"torch={torch.__version__}",
    f"transformers={transformers.__version__}",
    f"datasets={datasets.__version__}",
    "gpus=8",
)
PY

echo '=== validate the four distinct checkpoint-10000 inputs ==='
"$TASK_EVAL_PYTHON" - \
    "$TASK_NESTED_CKPT" "$TASK_GROUPREDUCE_CKPT" \
    "$TASK_DENSE_CKPT" "$TASK_DENSE_DDP_DEFAULT_CKPT" <<'PY'
import json
import os
import sys

import torch

from compositional.loading import is_compositional

nested, groupreduce, dense, dense_ddp = sys.argv[1:]
checkpoints = (nested, groupreduce, dense, dense_ddp)
assert len(set(map(os.path.realpath, checkpoints))) == 4, checkpoints

for checkpoint in checkpoints:
    assert os.path.isdir(checkpoint), checkpoint
    for filename in (
        "config.json", "model.safetensors", "trainer_state.json",
        "optimizer.pt", "scheduler.pt",
        "rng_state_0.pth", "rng_state_1.pth", "rng_state_2.pth",
        "rng_state_3.pth", "rng_state_4.pth", "rng_state_5.pth",
        "rng_state_6.pth", "rng_state_7.pth",
    ):
        path = os.path.join(checkpoint, filename)
        assert os.path.isfile(path) and os.path.getsize(path) > 0, path
    with open(os.path.join(checkpoint, "trainer_state.json"), encoding="utf-8") as handle:
        assert int(json.load(handle)["global_step"]) == 10000, checkpoint

expected = {
    nested: {
        "arm": "nested_ladder",
        "nested_tier_ranks": "64,128,320,512",
        "nested_tier_populations": "151936,32768,8192,2048",
    },
    groupreduce: {
        "arm": "groupreduce",
        "groupreduce_ranks": "1024,512,192,64",
        "groupreduce_populations": "2048,6144,24576,119168",
    },
}
for checkpoint, fields in expected.items():
    assert is_compositional(checkpoint), checkpoint
    embedding_path = os.path.join(checkpoint, "embedding.pt")
    assert os.path.isfile(embedding_path) and os.path.getsize(embedding_path) > 0
    assert not os.path.exists(os.path.join(checkpoint, "output_head.pt"))
    root = os.path.dirname(checkpoint)
    with open(os.path.join(root, "train_config.json"), encoding="utf-8") as handle:
        comp = json.load(handle)["compositional"]
    assert comp["tie_output"] is True, comp
    for key, value in fields.items():
        assert comp.get(key) == value, (checkpoint, key, comp.get(key), value)
    state = torch.load(embedding_path, map_location="cpu", weights_only=True)
    assert state and all(torch.isfinite(value).all() for value in state.values()), checkpoint

for checkpoint in (dense, dense_ddp):
    assert not is_compositional(checkpoint), checkpoint
    assert not os.path.exists(os.path.join(checkpoint, "embedding.pt"))
    assert not os.path.exists(os.path.join(checkpoint, "output_head.pt"))
    with open(os.path.join(checkpoint, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    assert config.get("tie_word_embeddings") is True, (checkpoint, config.get("tie_word_embeddings"))

print("FOUR_CHECKPOINTS_OK step=10000 nested=exact_tied groupreduce=exact_tied dense_controls=2")
PY

echo '=== verify fresh outputs and offline evaluation inputs ==='
[[ ! -e "$TASK_EVAL_LAUNCH_LOG" ]] \
    || die "refusing to overwrite evaluation log: $TASK_EVAL_LAUNCH_LOG"
[[ ! -e "$TASK_FINETUNE_OUTPUT" ]] \
    || die "refusing to reuse finetune output: $TASK_FINETUNE_OUTPUT"
for TASK_CHECKPOINT in "${TASK_CHECKPOINTS[@]}"; do
    for TASK_ARTIFACT in eval.log eval_ppl.json eval_benchmarks.json; do
        [[ ! -e "$TASK_CHECKPOINT/$TASK_ARTIFACT" ]] \
            || die "refusing to overwrite eval artifact: $TASK_CHECKPOINT/$TASK_ARTIFACT"
    done
done
test -s "$TASK_MODEL_DIR/config.json"
test -s "$TASK_MODEL_DIR/tokenizer.json"
for TASK_LANGUAGE in en vi zh ru de ar; do
    test -d "$TASK_EVAL_DIR/$TASK_LANGUAGE"
    test "$(find "$TASK_EVAL_DIR/$TASK_LANGUAGE" -type f -name '*.arrow' | wc -l)" -gt 0
done
for TASK_RELPATH in \
    facebook/xnli facebook/belebele cambridgeltl/xcopa \
    juletxara/xstory_cloze google-research-datasets/paws-x \
    Rowan/hellaswag alexandrainst/m_hellaswag; do
    test -d "$TASK_BENCH_ROOT/$TASK_RELPATH" \
        || die "missing offline eval dataset: $TASK_RELPATH"
done
"$TASK_EVAL_PYTHON" - <<'PY'
import os
from eval.benchmarks import TASK_CONFIGS, patch_lm_eval_dataset_paths

patch_lm_eval_dataset_paths(os.environ["LM_EVAL_DATASET_ROOT"])
tasks = [task for group in TASK_CONFIGS.values() for task in group]
assert len(tasks) == len(set(tasks)) == 26, tasks
print("OFFLINE_EVAL_INPUTS_OK tasks=26")
PY

TASK_AVAILABLE_BYTES=$(df -PB1 "$TASK_OUTPUT_BASE" | awk 'NR == 2 {print $4}')
[[ "$TASK_AVAILABLE_BYTES" =~ ^[0-9]+$ ]] || die 'could not determine free storage'
(( TASK_AVAILABLE_BYTES >= 100000000000 )) \
    || die "less than 100 GB available for four-model eval/finetune: $TASK_AVAILABLE_BYTES bytes"
echo "STORAGE_PREFLIGHT_OK available_bytes=$TASK_AVAILABLE_BYTES"

echo '=== verify all eight B200 GPUs are free before eval ==='
require_free_gpus 'BEFORE PRE-EVAL ACCELERATE CONFIG COPY'

echo '=== copy and verify the exact eight-GPU Accelerate config before eval ==='
install_accelerate_config

echo '=== run one eval_parallel.py command for all four checkpoints ==='
"$TASK_EVAL_PYTHON" -u eval/eval_parallel.py \
    --checkpoints "${TASK_CHECKPOINTS[@]}" \
    --eval-dir "$TASK_EVAL_DIR" \
    --tokenizer-name "$TASK_MODEL_DIR" \
    --bf16 \
    --num-gpus 8 \
    --log "$TASK_EVAL_LAUNCH_LOG"

echo '=== validate all four complete evaluations ==='
"$TASK_EVAL_PYTHON" - "${TASK_CHECKPOINTS[@]}" <<'PY'
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
        assert accuracy is not None and math.isfinite(float(accuracy)), (checkpoint, task, metrics)
    print(f"EVAL_JSON_OK checkpoint={checkpoint} ppl_languages=6 benchmark_tasks=26")
PY
grep -F 'All 4 evaluations done' "$TASK_EVAL_LAUNCH_LOG"
grep -F 'Loaded compositional model: arm=nested_ladder' "$TASK_NESTED_CKPT/eval.log"
grep -F 'Loaded compositional model: arm=groupreduce' "$TASK_GROUPREDUCE_CKPT/eval.log"
for TASK_DENSE_LOG in "$TASK_DENSE_CKPT/eval.log" "$TASK_DENSE_DDP_DEFAULT_CKPT/eval.log"; do
    if grep -Fq 'Loaded compositional model:' "$TASK_DENSE_LOG"; then
        die "dense checkpoint incorrectly routed through compositional loader: $TASK_DENSE_LOG"
    fi
done
TASK_EVAL_LOGS=(
    "$TASK_NESTED_CKPT/eval.log"
    "$TASK_GROUPREDUCE_CKPT/eval.log"
    "$TASK_DENSE_CKPT/eval.log"
    "$TASK_DENSE_DDP_DEFAULT_CKPT/eval.log"
    "$TASK_EVAL_LAUNCH_LOG"
)
scan_fatal_logs evaluation "${TASK_EVAL_LOGS[@]}"

echo '=== wait 60 seconds after eval and verify all GPUs free ==='
sleep 60
require_free_gpus '60 SECONDS AFTER FOUR-MODEL EVAL'

echo '=== copy and verify the exact eight-GPU Accelerate config ==='
install_accelerate_config

echo '=== wait 60 seconds after config copy and verify GPUs again ==='
sleep 60
require_free_gpus '60 SECONDS AFTER ACCELERATE CONFIG COPY AND BEFORE FINETUNE'

echo '=== standard four-model finetune preflight ==='
for TASK_RELPATH in \
    Rowan/hellaswag allenai/ai2_arc facebook/xnli \
    alexandrainst/m_arc alexandrainst/m_hellaswag \
    facebook/belebele cambridgeltl/xcopa \
    juletxara/xstory_cloze google-research-datasets/paws-x; do
    test -d "$TASK_BENCH_ROOT/$TASK_RELPATH" \
        || die "missing offline finetune dataset: $TASK_RELPATH"
done
echo 'PROTOCOL checkpoints=4 tasks=3 seeds=3 jobs=36 epochs=3 bf16 max_parallel_jobs=8'

echo '=== run one 36-job finetune queue across all eight GPUs ==='
"$TASK_EVAL_PYTHON" -u finetune/run_all.py \
    --checkpoints \
        nested_ladder_tied_t4="$TASK_NESTED_CKPT" \
        groupreduce_matched_nested_tied_t4="$TASK_GROUPREDUCE_CKPT" \
        dense_tied_baseline_b200="$TASK_DENSE_CKPT" \
        dense_tied_baseline_b200_ddp_default="$TASK_DENSE_DDP_DEFAULT_CKPT" \
    --tasks hellaswag arc_easy xnli \
    --seeds 42 123 456 \
    --tokenizer-name "$TASK_MODEL_DIR" \
    --num-gpus 8 \
    --output-dir "$TASK_FINETUNE_OUTPUT"

echo '=== validate all 36 finetune jobs and their summary ==='
"$TASK_EVAL_PYTHON" - \
    "$TASK_FINETUNE_OUTPUT" \
    "$TASK_NESTED_CKPT" "$TASK_GROUPREDUCE_CKPT" \
    "$TASK_DENSE_CKPT" "$TASK_DENSE_DDP_DEFAULT_CKPT" <<'PY'
import json
import math
import os
import sys

output_dir, nested, groupreduce, dense, dense_ddp = sys.argv[1:]
arms = {
    "nested_ladder_tied_t4": nested,
    "groupreduce_matched_nested_tied_t4": groupreduce,
    "dense_tied_baseline_b200": dense,
    "dense_tied_baseline_b200_ddp_default": dense_ddp,
}
expected_eval_tasks = {
    "hellaswag": {"hellaswag", "hellaswag_ar", "hellaswag_de", "hellaswag_ru", "hellaswag_vi"},
    "arc_easy": {"arc_easy", "arc_ar", "arc_de", "arc_ru", "arc_vi", "arc_zh"},
    "xnli": {"xnli_en", "xnli_vi", "xnli_zh", "xnli_de", "xnli_ru", "xnli_ar"},
}
validated = 0
for task, expected_tasks in expected_eval_tasks.items():
    for arm, checkpoint in arms.items():
        for seed in (42, 123, 456):
            stem = f"{task}_{arm}_seed{seed}"
            result_path = os.path.join(output_dir, stem + ".json")
            log_path = os.path.join(output_dir, stem + ".log")
            model_path = os.path.join(output_dir, "models", stem, "model_state.pt")
            for path in (result_path, log_path, model_path):
                assert os.path.isfile(path) and os.path.getsize(path) > 0, path
            with open(result_path, encoding="utf-8") as handle:
                result = json.load(handle)
            assert result["checkpoint"] == checkpoint, result["checkpoint"]
            assert result["task"] == task, result["task"]
            assert int(result["seed"]) == seed, result["seed"]
            assert int(result["epochs"]) == 3, result["epochs"]
            assert set(result["eval_results"]) == expected_tasks, result["eval_results"].keys()
            assert math.isfinite(float(result["train_time_s"])), result["train_time_s"]
            for eval_task, metrics in result["eval_results"].items():
                assert metrics.get("acc") is not None, (result_path, eval_task, metrics)
                assert math.isfinite(float(metrics["acc"])), (result_path, eval_task, metrics)
                if metrics.get("acc_norm") is not None:
                    assert math.isfinite(float(metrics["acc_norm"])), (result_path, eval_task, metrics)
            validated += 1
assert validated == 36, validated
summary = os.path.join(output_dir, "summary.md")
assert os.path.isfile(summary) and os.path.getsize(summary) > 0, summary
print("FOUR_MODEL_FINETUNE_OK jobs=36 tasks=3 seeds=3 epochs=3")
PY
mapfile -t TASK_FINETUNE_LOGS < <(
    find "$TASK_FINETUNE_OUTPUT" -maxdepth 1 -type f -name '*.log' | sort
)
[[ "${#TASK_FINETUNE_LOGS[@]}" -eq 36 ]] \
    || die "expected 36 finetune logs, found ${#TASK_FINETUNE_LOGS[@]}"
for TASK_LOG in "$TASK_FINETUNE_OUTPUT"/*_nested_ladder_tied_t4_seed*.log; do
    grep -Fq 'Loaded compositional model: arm=nested_ladder' "$TASK_LOG" \
        || die "Nested-Ladder loader confirmation missing from $TASK_LOG"
done
for TASK_LOG in "$TASK_FINETUNE_OUTPUT"/*_groupreduce_matched_nested_tied_t4_seed*.log; do
    grep -Fq 'Loaded compositional model: arm=groupreduce' "$TASK_LOG" \
        || die "GroupReduce loader confirmation missing from $TASK_LOG"
done
for TASK_LOG in \
    "$TASK_FINETUNE_OUTPUT"/*_dense_tied_baseline_b200_seed*.log \
    "$TASK_FINETUNE_OUTPUT"/*_dense_tied_baseline_b200_ddp_default_seed*.log; do
    if grep -Fq 'Loaded compositional model:' "$TASK_LOG"; then
        die "dense finetune incorrectly used compositional loader: $TASK_LOG"
    fi
done
scan_fatal_logs finetune "${TASK_FINETUNE_LOGS[@]}"

echo '=== wait 60 seconds after finetune and verify all GPUs free ==='
sleep 60
require_free_gpus '60 SECONDS AFTER FOUR-MODEL FINETUNE'
cat "$TASK_FINETUNE_OUTPUT/summary.md"

echo '=== launch and supervise one project burn worker per B200 ==='
TASK_CHILD_PIDS=()
cleanup_burn_workers() {
    trap - EXIT INT TERM
    if [[ "${#TASK_CHILD_PIDS[@]}" -gt 0 ]]; then
        kill -TERM "${TASK_CHILD_PIDS[@]}" 2>/dev/null || true
        wait "${TASK_CHILD_PIDS[@]}" 2>/dev/null || true
    fi
}
trap cleanup_burn_workers EXIT INT TERM

for TASK_GPU in 0 1 2 3 4 5 6 7; do
    TASK_BURN_LOG="/tmp/project_gpu_burn_four_model_workflow_gpu${TASK_GPU}.log"
    env CUDA_VISIBLE_DEVICES="$TASK_GPU" "$TASK_EVAL_PYTHON" -u "$TASK_BURN_SCRIPT" \
        >"$TASK_BURN_LOG" 2>&1 &
    TASK_CHILD_PIDS+=("$!")
    echo "launched_burn gpu=$TASK_GPU pid=$! log=$TASK_BURN_LOG"
done

sleep 30
declare -A TASK_EXPECTED_BURN_PIDS=()
for TASK_PID in "${TASK_CHILD_PIDS[@]}"; do
    kill -0 "$TASK_PID" || die "burn worker exited early: $TASK_PID"
    TASK_EXPECTED_BURN_PIDS["$TASK_PID"]=1
done
mapfile -t TASK_BURN_GPU_PIDS < <(gpu_pids)
[[ "${#TASK_BURN_GPU_PIDS[@]}" -eq 8 ]] \
    || die "expected 8 burn GPU processes, found ${#TASK_BURN_GPU_PIDS[@]}"
for TASK_PID in "${TASK_BURN_GPU_PIDS[@]}"; do
    [[ -n "${TASK_EXPECTED_BURN_PIDS[$TASK_PID]:-}" ]] \
        || die "unexpected final GPU process: $TASK_PID"
    TASK_CMDLINE="$(tr '\0' ' ' < "/proc/$TASK_PID/cmdline")"
    [[ "$TASK_CMDLINE" == *scripts/gpu_burn.py* ]] \
        || die "GPU PID is not project burn: $TASK_PID $TASK_CMDLINE"
done
declare -A TASK_BURNS_PER_UUID=()
while IFS=',' read -r TASK_GPU_UUID TASK_PID; do
    TASK_GPU_UUID="${TASK_GPU_UUID//[[:space:]]/}"
    TASK_PID="${TASK_PID//[[:space:]]/}"
    [[ -n "$TASK_GPU_UUID" && -n "$TASK_PID" ]] || continue
    TASK_BURNS_PER_UUID["$TASK_GPU_UUID"]=$((
        ${TASK_BURNS_PER_UUID["$TASK_GPU_UUID"]:-0} + 1
    ))
done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits)
mapfile -t TASK_GPU_UUIDS < <(
    nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits \
        | sed 's/[[:space:]]//g'
)
[[ "${#TASK_GPU_UUIDS[@]}" -eq 8 ]] || die 'expected 8 GPU UUIDs'
for TASK_GPU_UUID in "${TASK_GPU_UUIDS[@]}"; do
    [[ "${TASK_BURNS_PER_UUID[$TASK_GPU_UUID]:-0}" -eq 1 ]] \
        || die "expected one burn on $TASK_GPU_UUID, found ${TASK_BURNS_PER_UUID[$TASK_GPU_UUID]:-0}"
done
for TASK_GPU in 0 1 2 3 4 5 6 7; do
    grep -Fq 'gpu_burn_ready' "/tmp/project_gpu_burn_four_model_workflow_gpu${TASK_GPU}.log" \
        || die "burn readiness marker missing for GPU $TASK_GPU"
done
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,power.draw \
    --format=csv,noheader
echo 'TH2 FOUR-MODEL EVAL AND FINETUNE COMPLETE; GPU BURNS VERIFIED ON ALL 8 GPUS'
echo '=== supervising burns; workflow intentionally remains active ==='
set +e
wait -n "${TASK_CHILD_PIDS[@]}"
TASK_FIRST_EXIT=$?
set -e
die "a burn worker exited with status $TASK_FIRST_EXIT"
