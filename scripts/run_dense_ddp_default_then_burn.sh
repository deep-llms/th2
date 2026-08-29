#!/usr/bin/env bash
# Train the dense tied B200 baseline with the Transformers DDP default, stop at
# checkpoint 10k, wait one minute, require all GPUs free, then supervise burns.

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
        || die "GPU compute processes remain $stage: ${TASK_STAGE_GPU_PIDS[*]}"
    echo "ALL 8 B200 GPUS FREE $stage"
}

TASK_PROJECT_DIR="${SPARSE_EMB_PROJECT_DIR:?SPARSE_EMB_PROJECT_DIR is required}"
TASK_CONDA="${SPARSE_EMB_CONDA:?SPARSE_EMB_CONDA is required}"
TASK_PYTHON="${SPARSE_EMB_PYTHON:?SPARSE_EMB_PYTHON is required}"
TASK_MODEL_DIR="${SPARSE_EMB_MODEL_DIR:?SPARSE_EMB_MODEL_DIR is required}"
TASK_DATA_DIR="${SPARSE_EMB_DATA_DIR:?SPARSE_EMB_DATA_DIR is required}"
TASK_OUTPUT_BASE="${SPARSE_EMB_OUTPUT_BASE:?SPARSE_EMB_OUTPUT_BASE is required}"
TASK_OUTPUT_DIR="$TASK_OUTPUT_BASE/dense_tied_baseline_b200_ddp_default"
TASK_CHECKPOINT="$TASK_OUTPUT_DIR/checkpoint-10000"
TASK_REFERENCE_CHECKPOINT="$TASK_OUTPUT_BASE/dense_tied_baseline_b200/checkpoint-10000"
TASK_LOG_DIR="$TASK_OUTPUT_BASE/logs/dense_tied_baseline_b200_ddp_default_20260829"
TASK_EXPERIMENT_LOG="$TASK_LOG_DIR/experiments.log"
TASK_TRAIN_LOG="$TASK_LOG_DIR/dense_tied_baseline_b200_ddp_default.log"
TASK_TRAIN_SCRIPT="$TASK_PROJECT_DIR/scripts/train_dense_tied_baseline_b200_ddp_default.sh"
TASK_ACCELERATE_SOURCE="$TASK_PROJECT_DIR/resources/accelerate_config.yaml"
TASK_ACCELERATE_TARGET=/mnt/local/.cache/huggingface/accelerate/default_config.yaml
TASK_BURN_SCRIPT="$TASK_PROJECT_DIR/scripts/gpu_burn.py"
TASK_LOCK_DIR="$TASK_OUTPUT_BASE/.locks"
TASK_LOCK_FILE="$TASK_LOCK_DIR/dense_ddp_default_then_burn.lock"

mkdir -p "$TASK_LOCK_DIR"
exec 9>"$TASK_LOCK_FILE"
if ! flock -n 9; then
    echo "WORKFLOW_ALREADY_RUNNING lock=$TASK_LOCK_FILE"
    exit 0
fi

echo '=== dense tied DDP-default checkpoint-10k -> burn workflow ==='
date -u
hostname
cd "$TASK_PROJECT_DIR"

test -x "$TASK_CONDA" || die "missing conda: $TASK_CONDA"
test -x "$TASK_PYTHON" || die "missing sparse_emb Python: $TASK_PYTHON"
test -s "$TASK_TRAIN_SCRIPT" || die "missing training launcher: $TASK_TRAIN_SCRIPT"
test -s "$TASK_ACCELERATE_SOURCE" || die "missing Accelerate config"
test -s "$TASK_BURN_SCRIPT" || die "missing project GPU-burn script"
test -s "$TASK_MODEL_DIR/config.json" || die "missing model config"
test -s "$TASK_MODEL_DIR/tokenizer.json" || die "missing tokenizer"
test -d "$TASK_DATA_DIR" || die "missing training data"
test -d "$TASK_REFERENCE_CHECKPOINT" || die "missing matched reference checkpoint"

case "$TASK_OUTPUT_DIR" in
    /mnt/local/_outputs/*/dense_tied_baseline_b200_ddp_default) ;;
    *) die "unexpected output path: $TASK_OUTPUT_DIR" ;;
esac
[[ ! -e "$TASK_OUTPUT_DIR" ]] || die "fresh output already exists: $TASK_OUTPUT_DIR"
[[ ! -e "$TASK_LOG_DIR" ]] || die "fresh log directory already exists: $TASK_LOG_DIR"
if grep -Eq '^[[:space:]]*--ddp_find_unused_parameters([[:space:]]|$)' "$TASK_TRAIN_SCRIPT"; then
    die 'DDP-default launcher unexpectedly passes --ddp_find_unused_parameters'
fi

TASK_AVAILABLE_BYTES="$(df -PB1 "$TASK_OUTPUT_BASE" | awk 'NR == 2 {print $4}')"
[[ "$TASK_AVAILABLE_BYTES" =~ ^[0-9]+$ ]] || die 'could not determine free disk space'
(( TASK_AVAILABLE_BYTES >= 200000000000 )) \
    || die "less than 200 GB free for the baseline checkpoint series: $TASK_AVAILABLE_BYTES"
echo "available_output_bytes=$TASK_AVAILABLE_BYTES"

echo '=== activate and validate sparse_emb ==='
eval "$("$TASK_CONDA" shell.bash hook)"
conda activate sparse_emb
[[ "${CONDA_DEFAULT_ENV:-}" == sparse_emb ]] || die 'failed to activate sparse_emb'
[[ "$(command -v python3.11)" == "$TASK_PYTHON" ]] \
    || die "wrong Python after activation: $(command -v python3.11)"
"$TASK_PYTHON" - <<'PY'
import accelerate
import datasets
import torch
import transformers
from transformers import TrainingArguments

field = TrainingArguments.__dataclass_fields__["ddp_find_unused_parameters"]
assert field.default is None, field.default
print(
    "TRAINING_ENV_OK",
    f"torch={torch.__version__}",
    f"transformers={transformers.__version__}",
    f"datasets={datasets.__version__}",
    f"accelerate={accelerate.__version__}",
    "ddp_find_unused_parameters_cli_default=None",
)
PY

echo '=== verify native tied model and sampled inputs ==='
"$TASK_PYTHON" - "$TASK_MODEL_DIR/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)
assert config.get("model_type") == "qwen3", config.get("model_type")
assert config.get("tie_word_embeddings") is True, config.get("tie_word_embeddings")
assert config.get("hidden_size") == 1024, config.get("hidden_size")
assert config.get("vocab_size") == 151936, config.get("vocab_size")
print("QWEN3_0.6B_NATIVE_TIED_CONFIG_OK")
PY
for TASK_LANG in en vi zh ru de ar; do
    test -d "$TASK_DATA_DIR/$TASK_LANG" || die "missing training language: $TASK_LANG"
    TASK_ARROW_COUNT="$(find "$TASK_DATA_DIR/$TASK_LANG" -type f -name '*.arrow' \
        ! -name 'cache-*' | wc -l)"
    [[ "$TASK_ARROW_COUNT" -gt 0 ]] || die "no source Arrow data for $TASK_LANG"
    echo "source_language=$TASK_LANG arrow_files=$TASK_ARROW_COUNT"
done

echo '=== install exact eight-GPU bf16 Accelerate configuration ==='
mkdir -p "$(dirname "$TASK_ACCELERATE_TARGET")"
cp "$TASK_ACCELERATE_SOURCE" "$TASK_ACCELERATE_TARGET"
cmp "$TASK_ACCELERATE_SOURCE" "$TASK_ACCELERATE_TARGET"
grep -Fxq 'distributed_type: MULTI_GPU' "$TASK_ACCELERATE_TARGET"
grep -Fxq 'mixed_precision: bf16' "$TASK_ACCELERATE_TARGET"
grep -Fxq 'num_processes: 8' "$TASK_ACCELERATE_TARGET"

echo '=== require all GPUs free before training ==='
validate_b200_node
require_free_gpus 'BEFORE DDP-DEFAULT TRAINING'

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo '=== resolve exact experiment definition ==='
TASK_EXPERIMENT_INDEX="$("$TASK_PYTHON" - "$TASK_OUTPUT_DIR" <<'PY'
import sys
from run_experiments import EXPERIMENT_COMMANDS

name = "dense_tied_baseline_b200_ddp_default"
matches = [(index, experiment) for index, experiment in enumerate(EXPERIMENT_COMMANDS)
           if experiment.get("name") == name]
assert len(matches) == 1, matches
index, experiment = matches[0]
assert experiment["cmd"] == "bash scripts/train_dense_tied_baseline_b200_ddp_default.sh", experiment
assert experiment["output_dir"] == sys.argv[1], (experiment["output_dir"], sys.argv[1])
assert experiment.get("require_fresh_output") is True, experiment
print(index)
PY
)"
[[ "$TASK_EXPERIMENT_INDEX" =~ ^[0-9]+$ ]] \
    || die "invalid experiment index: $TASK_EXPERIMENT_INDEX"
echo "experiment_index=$TASK_EXPERIMENT_INDEX"
echo 'only_intended_ablation=omit_ddp_find_unused_parameters'
echo 'architecture=native_Qwen3_dense_input_output_exactly_tied'
echo 'precision=bf16 seed=42 block_size=2048'
echo 'batch=16_per_device_x_4_accumulation_x_8_gpus=512_sequences_per_update'
echo 'schedule=original_one_epoch_with_runner_stop_at_checkpoint_10000'

echo '=== train DDP-default dense tied baseline to checkpoint 10000 ==='
"$TASK_PYTHON" -u run_experiments.py \
    --experiments "$TASK_EXPERIMENT_INDEX" \
    --stop-at-step 10000 \
    --log-dir "$TASK_LOG_DIR"

echo '=== verify successful checkpoint-10000 completion ==='
grep -Fq 'dense_tied_baseline_b200_ddp_default: STOPPED at step 10000' \
    "$TASK_EXPERIMENT_LOG" || die 'runner did not report a successful checkpoint-10k stop'
for TASK_FILE in \
    config.json model.safetensors trainer_state.json optimizer.pt scheduler.pt \
    rng_state_0.pth rng_state_1.pth rng_state_2.pth rng_state_3.pth \
    rng_state_4.pth rng_state_5.pth rng_state_6.pth rng_state_7.pth; do
    test -s "$TASK_CHECKPOINT/$TASK_FILE" \
        || die "missing or empty checkpoint artifact: $TASK_CHECKPOINT/$TASK_FILE"
done
"$TASK_PYTHON" - "$TASK_CHECKPOINT" <<'PY'
import json
import os
import sys

checkpoint = sys.argv[1]
with open(os.path.join(checkpoint, "trainer_state.json"), encoding="utf-8") as handle:
    trainer_state = json.load(handle)
assert int(trainer_state["global_step"]) == 10000, trainer_state["global_step"]
with open(os.path.join(checkpoint, "config.json"), encoding="utf-8") as handle:
    config = json.load(handle)
assert config.get("tie_word_embeddings") is True, config.get("tie_word_embeddings")
print("DENSE_DDP_DEFAULT_CHECKPOINT_OK step=10000 native_tied=true")
PY
if grep -HniE 'CUDA out of memory|OutOfMemoryError|NCCL.*(unhandled|system error|remote process exited|watchdog|timeout)|Segmentation fault|Bus error' \
        "$TASK_TRAIN_LOG" "$TASK_EXPERIMENT_LOG"; then
    die 'fatal signature found in training logs'
fi

echo '=== wait 60 seconds after training ==='
sleep 60

echo '=== prove training is gone and all GPUs are free before burn ==='
if pgrep -af '[t]rain.py'; then
    die 'train.py process remains after the 60-second guard'
fi
if pgrep -af '[r]un_experiments.py'; then
    die 'run_experiments.py process remains after the 60-second guard'
fi
if pgrep -af '[a]ccelerate.commands.launch|[a]ccelerate launch'; then
    die 'Accelerate launcher remains after the 60-second guard'
fi
validate_b200_node
require_free_gpus '60 SECONDS AFTER TRAINING AND BEFORE BURN'

echo '=== start one supervised project GPU-burn worker per B200 ==='
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
    TASK_BURN_LOG="/tmp/project_gpu_burn_ddp_default_gpu${TASK_GPU}.log"
    env CUDA_VISIBLE_DEVICES="$TASK_GPU" "$TASK_PYTHON" -u "$TASK_BURN_SCRIPT" \
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
    || die "expected 8 GPU-burn processes, found ${#TASK_BURN_GPU_PIDS[@]}"
for TASK_PID in "${TASK_BURN_GPU_PIDS[@]}"; do
    [[ -n "${TASK_EXPECTED_BURN_PIDS[$TASK_PID]:-}" ]] \
        || die "unexpected GPU process after burn launch: $TASK_PID"
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
    grep -Fq 'gpu_burn_ready' "/tmp/project_gpu_burn_ddp_default_gpu${TASK_GPU}.log" \
        || die "burn readiness marker missing for GPU $TASK_GPU"
done
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,power.draw \
    --format=csv,noheader
echo 'TH2 DDP-DEFAULT TRAINING COMPLETE; GPU BURN VERIFIED ON ALL 8 GPUS'
echo '=== supervising burns; this workflow intentionally remains active ==='
set +e
wait -n "${TASK_CHILD_PIDS[@]}"
TASK_FIRST_EXIT=$?
set -e
die "a burn worker exited with status $TASK_FIRST_EXIT"
