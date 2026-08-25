#!/usr/bin/env bash
# Pure-local G16 R128 checkpoint-10k handoff for the th2 8xB200 node.
#
# This workflow is intentionally fail-closed. It waits for the exact training
# workers to exit at a complete checkpoint-10000, runs and validates full eval,
# runs and validates the standard nine-job finetune battery, and starts GPU
# burns only after every preceding stage succeeds.
set -euo pipefail

TASK_PROJECT_DIR="${SPARSE_EMB_PROJECT_DIR:?SPARSE_EMB_PROJECT_DIR is required}"
TASK_OUTPUT_BASE="${SPARSE_EMB_OUTPUT_BASE:?SPARSE_EMB_OUTPUT_BASE is required}"
TASK_MODEL_DIR="${SPARSE_EMB_MODEL_DIR:?SPARSE_EMB_MODEL_DIR is required}"
TASK_EVAL_DIR="${SPARSE_EMB_EVAL_DIR:?SPARSE_EMB_EVAL_DIR is required}"
TASK_BENCH_ROOT="${SPARSE_EMB_BENCH_ROOT:?SPARSE_EMB_BENCH_ROOT is required}"
TASK_EVAL_PYTHON="${SPARSE_EMB_EVAL_PYTHON:?SPARSE_EMB_EVAL_PYTHON is required}"

TASK_TRAIN_OUTPUT="$TASK_OUTPUT_BASE/pure_local_tied_g16_r128"
TASK_CHECKPOINT="$TASK_TRAIN_OUTPUT/checkpoint-10000"
TASK_TRAIN_LOG_DIR="$TASK_OUTPUT_BASE/logs/pure_local_tied_g16_r128_20260825"
TASK_EXPERIMENT_LOG="$TASK_TRAIN_LOG_DIR/experiments.log"
TASK_TRAIN_LOG="$TASK_TRAIN_LOG_DIR/pure_local_tied_g16_r128.log"
TASK_EVAL_LAUNCH_LOG="$TASK_OUTPUT_BASE/eval_parallel_pure_local_tied_g16_r128_10k_20260825.log"
TASK_FINETUNE_OUTPUT="$TASK_OUTPUT_BASE/finetune_pure_local_tied_g16_r128_10k_20260825"
TASK_BURN_SCRIPT="$TASK_PROJECT_DIR/scripts/gpu_burn.py"

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
    nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
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

scan_training_fatal_logs() {
    local pattern
    # run_experiments.py intentionally terminates Accelerate with SIGTERM after
    # checkpoint-10000 is safely written, so its resulting SignalException and
    # traceback are expected.  Scan only signatures that are never part of that
    # controlled stop; checkpoint completeness and runner status are validated
    # independently below.
    pattern='CUDA out of memory|OutOfMemoryError|NCCL.*(unhandled|system error|remote process exited|watchdog|timeout)|Segmentation fault|Bus error'
    if grep -HniE "$pattern" "$TASK_TRAIN_LOG" "$TASK_EXPERIMENT_LOG"; then
        die 'fatal signature found in training logs'
    fi
}

echo '=== Pure-local 10k -> eval -> finetune -> burn handoff ==='
date -u
hostname
cd "$TASK_PROJECT_DIR"
test -x "$TASK_EVAL_PYTHON"
test -s "$TASK_BURN_SCRIPT"
validate_b200_node
[[ ! -e "$TASK_EVAL_LAUNCH_LOG" ]] \
    || die "refusing to overwrite evaluation log: $TASK_EVAL_LAUNCH_LOG"
[[ ! -e "$TASK_FINETUNE_OUTPUT" ]] \
    || die "refusing to reuse finetune output: $TASK_FINETUNE_OUTPUT"

echo '=== wait for exact Pure-local training to stop at checkpoint-10000 ==='
TASK_TRAINING_DONE=0
for TASK_POLL in $(seq 1 720); do
    # Take one process snapshot, then classify every PID from that snapshot.
    # This avoids comparing two nvidia-smi calls across a rank-shutdown race.
    mapfile -t TASK_ACTIVE_GPU_PIDS < <(gpu_pids)
    TASK_TRAIN_PIDS=()
    TASK_UNEXPECTED_GPU_PIDS=()
    TASK_TRANSIENT_GPU_PIDS=()
    for TASK_PID in "${TASK_ACTIVE_GPU_PIDS[@]}"; do
        if [[ ! -r "/proc/$TASK_PID/cmdline" ]]; then
            TASK_TRANSIENT_GPU_PIDS+=("$TASK_PID")
            continue
        fi
        TASK_CMDLINE=$(tr '\0' ' ' < "/proc/$TASK_PID/cmdline" 2>/dev/null || true)
        if [[ "$TASK_CMDLINE" == *train_compositional.py* \
                && "$TASK_CMDLINE" == *"--output_dir $TASK_TRAIN_OUTPUT"* \
                && "$TASK_CMDLINE" == *"--arm pure_local"* \
                && "$TASK_CMDLINE" == *"--pure_local_rank 128"* \
                && "$TASK_CMDLINE" == *"--num_groups 16"* \
                && "$TASK_CMDLINE" == *"--tie_output"* ]]; then
            TASK_TRAIN_PIDS+=("$TASK_PID")
        else
            echo "unexpected_gpu_pid=$TASK_PID cmd=$TASK_CMDLINE"
            TASK_UNEXPECTED_GPU_PIDS+=("$TASK_PID")
        fi
    done

    [[ "${#TASK_UNEXPECTED_GPU_PIDS[@]}" -eq 0 ]] \
        || die "unexpected GPU processes during training: ${TASK_UNEXPECTED_GPU_PIDS[*]}"
    if [[ "${#TASK_TRANSIENT_GPU_PIDS[@]}" -gt 0 ]]; then
        if [[ -d "$TASK_CHECKPOINT" ]]; then
            echo "checkpoint-10000 rank cleanup in progress; stale GPU PIDs=${TASK_TRANSIENT_GPU_PIDS[*]}"
            sleep 10
            continue
        fi
        die "unreadable GPU processes before checkpoint-10000: ${TASK_TRANSIENT_GPU_PIDS[*]}"
    fi

    if [[ "${#TASK_TRAIN_PIDS[@]}" -gt 0 ]]; then
        if [[ "${#TASK_TRAIN_PIDS[@]}" -ne 8 ]]; then
            if [[ -d "$TASK_CHECKPOINT" ]]; then
                echo "checkpoint-10000 controlled rank shutdown in progress; workers=${#TASK_TRAIN_PIDS[@]} pids=${TASK_TRAIN_PIDS[*]}"
                sleep 10
                continue
            fi
            die "expected 8 Pure-local workers before checkpoint-10000, found ${#TASK_TRAIN_PIDS[@]}: ${TASK_TRAIN_PIDS[*]}"
        fi
        if (( TASK_POLL == 1 || TASK_POLL % 10 == 0 )); then
            TASK_LATEST_STEP=$(find "$TASK_TRAIN_OUTPUT" -mindepth 1 -maxdepth 1 \
                -type d -name 'checkpoint-*' -printf '%f\n' 2>/dev/null \
                | sed 's/checkpoint-//' | sort -n | tail -1)
            echo "training_poll=$TASK_POLL workers=8 latest_checkpoint=${TASK_LATEST_STEP:-none}"
            tail -c 120000 "$TASK_TRAIN_LOG" 2>/dev/null \
                | tr '\r' '\n' | grep -E "\{'loss':" | tail -1 || true
            nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
        fi
        sleep 60
        continue
    fi

    [[ "${#TASK_ACTIVE_GPU_PIDS[@]}" -eq 0 ]] \
        || die "training workers vanished but other GPU processes remain: ${TASK_ACTIVE_GPU_PIDS[*]}"
    [[ -d "$TASK_CHECKPOINT" ]] \
        || die 'Pure-local training stopped before checkpoint-10000 existed'
    echo "training_workers=0 checkpoint_present=$TASK_CHECKPOINT poll=$TASK_POLL"
    TASK_TRAINING_DONE=1
    break
done
[[ "$TASK_TRAINING_DONE" -eq 1 ]] \
    || die 'Pure-local training did not stop successfully within 12 hours'

echo '=== verify exact complete tied Pure-local checkpoint ==='
for TASK_FILE in \
    config.json model.safetensors trainer_state.json optimizer.pt scheduler.pt \
    embedding.pt rng_state_0.pth rng_state_1.pth rng_state_2.pth \
    rng_state_3.pth rng_state_4.pth rng_state_5.pth rng_state_6.pth \
    rng_state_7.pth; do
    test -s "$TASK_CHECKPOINT/$TASK_FILE" \
        || die "missing or empty checkpoint artifact: $TASK_CHECKPOINT/$TASK_FILE"
done
test -s "$TASK_TRAIN_OUTPUT/train_config.json"
[[ ! -e "$TASK_CHECKPOINT/output_head.pt" ]] \
    || die 'exactly tied Pure-local checkpoint unexpectedly has output_head.pt'

"$TASK_EVAL_PYTHON" - "$TASK_CHECKPOINT" "$TASK_TRAIN_OUTPUT/train_config.json" <<'PY'
import json
import math
import os
import sys

import torch

checkpoint, train_config_path = sys.argv[1:]
with open(os.path.join(checkpoint, "trainer_state.json"), encoding="utf-8") as handle:
    trainer_state = json.load(handle)
assert int(trainer_state["global_step"]) == 10000, trainer_state["global_step"]
with open(train_config_path, encoding="utf-8") as handle:
    train_config = json.load(handle)
comp = train_config["compositional"]
assert comp["arm"] == "pure_local", comp
assert int(comp["pure_local_rank"]) == 128, comp
assert int(comp["num_groups"]) == 16, comp
assert comp["tie_output"] is True, comp
assert not comp.get("independent_lowrank_output", False), comp
state = torch.load(os.path.join(checkpoint, "embedding.pt"), map_location="cpu", weights_only=True)
assert set(state) == {"token_factors", "local_weight", "bias"}, state.keys()
assert tuple(state["token_factors"].shape) == (16, 9496, 128)
assert tuple(state["local_weight"].shape) == (16, 1024, 128)
assert tuple(state["bias"].shape) == (1024,)
for name, tensor in state.items():
    assert torch.isfinite(tensor).all(), name
print("PURE_LOCAL_CHECKPOINT_OK step=10000 rank=128 groups=16 exact_tied_output=true")
PY

for TASK_RETRY in $(seq 1 12); do
    if grep -Fq 'pure_local_tied_g16_r128: STOPPED at step 10000' "$TASK_EXPERIMENT_LOG" 2>/dev/null; then
        break
    fi
    sleep 5
done
grep -F 'pure_local_tied_g16_r128: STOPPED at step 10000' "$TASK_EXPERIMENT_LOG"
scan_training_fatal_logs
require_free_gpus 'AFTER TRAINING AND BEFORE EVAL'

echo '=== verify offline eval inputs and patch lm-eval paths ==='
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
for TASK_ARTIFACT in eval.log eval_ppl.json eval_benchmarks.json; do
    [[ ! -e "$TASK_CHECKPOINT/$TASK_ARTIFACT" ]] \
        || die "refusing to overwrite eval artifact: $TASK_CHECKPOINT/$TASK_ARTIFACT"
done

echo '=== run full eval_parallel.py at Pure-local checkpoint-10000 ==='
"$TASK_EVAL_PYTHON" -u eval/eval_parallel.py \
    --checkpoints "$TASK_CHECKPOINT" \
    --eval-dir "$TASK_EVAL_DIR" \
    --tokenizer-name "$TASK_MODEL_DIR" \
    --bf16 \
    --num-gpus 8 \
    --log "$TASK_EVAL_LAUNCH_LOG"

echo '=== wait one minute, then validate eval and free GPUs ==='
sleep 60
"$TASK_EVAL_PYTHON" - "$TASK_CHECKPOINT" <<'PY'
import json
import math
import os
import sys

checkpoint = sys.argv[1]
with open(os.path.join(checkpoint, "eval_ppl.json"), encoding="utf-8") as handle:
    perplexity = json.load(handle)
with open(os.path.join(checkpoint, "eval_benchmarks.json"), encoding="utf-8") as handle:
    benchmarks = json.load(handle)
assert set(perplexity) == {"en", "vi", "zh", "ru", "de", "ar"}, perplexity.keys()
for language, metrics in perplexity.items():
    assert int(metrics["num_tokens"]) > 0, (language, metrics)
    assert math.isfinite(float(metrics["loss"])), (language, metrics)
    assert math.isfinite(float(metrics["perplexity"])), (language, metrics)
assert len(benchmarks) == 26, len(benchmarks)
for task, metrics in benchmarks.items():
    accuracy = metrics.get("acc,none", metrics.get("acc"))
    assert accuracy is not None and math.isfinite(float(accuracy)), (task, metrics)
print("PURE_LOCAL_EVAL_OK ppl_languages=6 benchmark_tasks=26")
PY
grep -F 'All 1 evaluations done' "$TASK_EVAL_LAUNCH_LOG"
grep -F 'Loaded compositional model: arm=pure_local' "$TASK_CHECKPOINT/eval.log"
scan_fatal_logs evaluation "$TASK_CHECKPOINT/eval.log" "$TASK_EVAL_LAUNCH_LOG"
require_free_gpus 'ONE MINUTE AFTER EVAL'

echo '=== standard Pure-local finetune preflight ==='
for TASK_RELPATH in \
    Rowan/hellaswag allenai/ai2_arc facebook/xnli \
    alexandrainst/m_arc alexandrainst/m_hellaswag \
    facebook/belebele cambridgeltl/xcopa \
    juletxara/xstory_cloze google-research-datasets/paws-x; do
    test -d "$TASK_BENCH_ROOT/$TASK_RELPATH" \
        || die "missing offline finetune dataset: $TASK_RELPATH"
done
TASK_AVAILABLE_BYTES=$(df -PB1 "$TASK_OUTPUT_BASE" | awk 'NR == 2 {print $4}')
[[ "$TASK_AVAILABLE_BYTES" =~ ^[0-9]+$ ]] \
    || die 'could not determine free storage'
(( TASK_AVAILABLE_BYTES >= 20000000000 )) \
    || die "less than 20 GB available for nine finetune jobs: $TASK_AVAILABLE_BYTES bytes"
echo "FINETUNE_PREFLIGHT_OK fresh_output=$TASK_FINETUNE_OUTPUT available_bytes=$TASK_AVAILABLE_BYTES"
echo 'PROTOCOL checkpoint=1 tasks=3 seeds=3 jobs=9 epochs=3 bf16 max_parallel_jobs=8'

echo '=== run the standard nine-job finetune battery ==='
"$TASK_EVAL_PYTHON" -u finetune/run_all.py \
    --checkpoints pure_local_tied_g16_r128="$TASK_CHECKPOINT" \
    --tasks hellaswag arc_easy xnli \
    --seeds 42 123 456 \
    --tokenizer-name "$TASK_MODEL_DIR" \
    --num-gpus 8 \
    --output-dir "$TASK_FINETUNE_OUTPUT"

echo '=== wait one minute, then validate finetune and free GPUs ==='
sleep 60
"$TASK_EVAL_PYTHON" - "$TASK_FINETUNE_OUTPUT" "$TASK_CHECKPOINT" <<'PY'
import json
import math
import os
import sys

output_dir, checkpoint = sys.argv[1:]
arm = "pure_local_tied_g16_r128"
expected_eval_tasks = {
    "hellaswag": {"hellaswag", "hellaswag_ar", "hellaswag_de", "hellaswag_ru", "hellaswag_vi"},
    "arc_easy": {"arc_easy", "arc_ar", "arc_de", "arc_ru", "arc_vi", "arc_zh"},
    "xnli": {"xnli_en", "xnli_vi", "xnli_zh", "xnli_de", "xnli_ru", "xnli_ar"},
}
validated = 0
for task, expected_tasks in expected_eval_tasks.items():
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
assert validated == 9, validated
summary = os.path.join(output_dir, "summary.md")
assert os.path.isfile(summary) and os.path.getsize(summary) > 0, summary
print("PURE_LOCAL_FINETUNE_OK jobs=9 tasks=3 seeds=3 epochs=3")
PY
mapfile -t TASK_FINETUNE_LOGS < <(find "$TASK_FINETUNE_OUTPUT" -maxdepth 1 -type f -name '*.log' | sort)
[[ "${#TASK_FINETUNE_LOGS[@]}" -eq 9 ]] \
    || die "expected 9 finetune logs, found ${#TASK_FINETUNE_LOGS[@]}"
for TASK_FINETUNE_LOG in "${TASK_FINETUNE_LOGS[@]}"; do
    grep -Fq 'Loaded compositional model: arm=pure_local' "$TASK_FINETUNE_LOG" \
        || die "Pure-local loader confirmation missing from $TASK_FINETUNE_LOG"
done
scan_fatal_logs finetune "${TASK_FINETUNE_LOGS[@]}"
require_free_gpus 'ONE MINUTE AFTER FINETUNE'
cat "$TASK_FINETUNE_OUTPUT/summary.md"

echo '=== start one supervised GPU-burn worker per B200 ==='
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
    TASK_BURN_LOG="/tmp/project_gpu_burn_gpu${TASK_GPU}.log"
    env CUDA_VISIBLE_DEVICES="$TASK_GPU" /usr/bin/python3 -u "$TASK_BURN_SCRIPT" \
        >"$TASK_BURN_LOG" 2>&1 &
    TASK_CHILD_PIDS+=("$!")
    echo "launched_burn gpu=$TASK_GPU pid=$! log=$TASK_BURN_LOG"
done
sleep 30
for TASK_PID in "${TASK_CHILD_PIDS[@]}"; do
    kill -0 "$TASK_PID" || die "burn worker exited early: $TASK_PID"
done
mapfile -t TASK_BURN_GPU_PIDS < <(gpu_pids)
[[ "${#TASK_BURN_GPU_PIDS[@]}" -eq 8 ]] \
    || die "expected 8 GPU-burn processes, found ${#TASK_BURN_GPU_PIDS[@]}"
for TASK_PID in "${TASK_BURN_GPU_PIDS[@]}"; do
    TASK_CMDLINE=$(tr '\0' ' ' < "/proc/$TASK_PID/cmdline")
    [[ "$TASK_CMDLINE" == *scripts/gpu_burn.py* ]] \
        || die "unexpected final GPU process $TASK_PID: $TASK_CMDLINE"
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
[[ "${#TASK_GPU_UUIDS[@]}" -eq 8 ]] \
    || die "expected 8 GPU UUIDs, found ${#TASK_GPU_UUIDS[@]}"
for TASK_GPU_UUID in "${TASK_GPU_UUIDS[@]}"; do
    [[ "${TASK_BURNS_PER_UUID[$TASK_GPU_UUID]:-0}" -eq 1 ]] \
        || die "expected one burn on GPU $TASK_GPU_UUID, found ${TASK_BURNS_PER_UUID[$TASK_GPU_UUID]:-0}"
done
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
echo 'TH2 FULL PURE-LOCAL WORKFLOW COMPLETE; GPU BURN VERIFIED ON ALL 8 GPUS'
echo '=== supervising burns; this workflow intentionally remains active ==='
set +e
wait -n "${TASK_CHILD_PIDS[@]}"
TASK_FIRST_EXIT=$?
set -e
die "a burn worker exited with status $TASK_FIRST_EXIT"
