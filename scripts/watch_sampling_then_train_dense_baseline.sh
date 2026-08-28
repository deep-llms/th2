#!/usr/bin/env bash
# Wait for the offline CulturaX sampler, verify its exact output, safely stop the
# runner-provided B200 burn, and launch the matched dense tied baseline.

set -euo pipefail

die() {
    echo "ERROR: $*" >&2
    exit 1
}

TASK_PROJECT_DIR="${SPARSE_EMB_PROJECT_DIR:-$PWD}"
TASK_CONDA="${SPARSE_EMB_CONDA:-/mnt/local/conda-py311/bin/conda}"
TASK_PYTHON="${SPARSE_EMB_PYTHON:-/mnt/local/conda-py311/envs/sparse_emb/bin/python3.11}"
TASK_MODEL_DIR="${SPARSE_EMB_MODEL_DIR:-/mnt/local/_models/@PROJECT@/Qwen3-0.6B}"
TASK_DATASET_ROOT="${SPARSE_EMB_DATASET_ROOT:-/mnt/local/_data/@PROJECT@/data/Qwen_Qwen3-0.6B}"
TASK_TRAIN_DATA="$TASK_DATASET_ROOT/train"
TASK_OUTPUT_BASE="${SPARSE_EMB_OUTPUT_BASE:-/mnt/local/_outputs/@PROJECT@}"
TASK_OUTPUT_DIR="$TASK_OUTPUT_BASE/dense_tied_baseline_b200"
TASK_LOG_DIR="${SPARSE_EMB_BASELINE_LOG_DIR:-$TASK_OUTPUT_BASE/logs/dense_tied_baseline_b200_fresh_b200_20260828}"
TASK_ACCELERATE_SOURCE="$TASK_PROJECT_DIR/resources/accelerate_config.yaml"
TASK_ACCELERATE_TARGET="/mnt/local/.cache/huggingface/accelerate/default_config.yaml"
TASK_LOCK_DIR="$TASK_OUTPUT_BASE/.locks"
TASK_LOCK_FILE="$TASK_LOCK_DIR/sample_to_dense_baseline.lock"

verify_fresh_training_paths() {
    [ ! -e "$TASK_OUTPUT_DIR" ] \
        || die "fresh baseline output already exists: $TASK_OUTPUT_DIR"
    [ ! -e "$TASK_LOG_DIR" ] \
        || die "fresh baseline log already exists: $TASK_LOG_DIR"
}

mkdir -p "$TASK_LOCK_DIR"
exec 9>"$TASK_LOCK_FILE"
if ! flock -n 9; then
    echo "HANDOFF_ALREADY_RUNNING lock=$TASK_LOCK_FILE"
    exit 0
fi

echo '=== th2 sampling-to-dense-baseline handoff ==='
date -u
hostname

test -x "$TASK_PYTHON" || die "missing sparse_emb Python: $TASK_PYTHON"
test -x "$TASK_CONDA" || die "missing conda: $TASK_CONDA"
test -s "$TASK_ACCELERATE_SOURCE" || die "missing Accelerate config: $TASK_ACCELERATE_SOURCE"
test -s "$TASK_MODEL_DIR/config.json" || die "missing local model config"
test -s "$TASK_MODEL_DIR/tokenizer.json" || die "missing local tokenizer"
test -s "$TASK_PROJECT_DIR/run_experiments.py" || die "missing run_experiments.py"
test -s "$TASK_PROJECT_DIR/scripts/train_dense_tied_baseline_b200.sh" \
    || die "missing dense baseline launcher"

# Refuse before touching any GPU if this fresh run is no longer fresh. This also
# protects a training job if the remote runner executes commands.sh more than once.
case "$TASK_OUTPUT_DIR" in
    /mnt/local/_outputs/*/dense_tied_baseline_b200) ;;
    *) die "unexpected output path: $TASK_OUTPUT_DIR" ;;
esac
case "$TASK_LOG_DIR" in
    /mnt/local/_outputs/*/logs/dense_tied_baseline_b200_*) ;;
    *) die "unexpected log path: $TASK_LOG_DIR" ;;
esac
verify_fresh_training_paths

echo '=== wait for the existing offline sampler ==='
TASK_WAIT_START="$SECONDS"
TASK_MAX_POLLS=720
for ((TASK_POLL = 1; TASK_POLL <= TASK_MAX_POLLS; TASK_POLL++)); do
    if ! pgrep -f '[p]repare_data.py sample' >/dev/null; then
        echo "sampler_process_gone wait_seconds=$((SECONDS - TASK_WAIT_START))"
        break
    fi
    TASK_EN_SHARDS="$(find "$TASK_TRAIN_DATA/en" -mindepth 1 -maxdepth 1 \
        -type d -name 'shard_*' 2>/dev/null | wc -l || true)"
    TASK_CURRENT_SIZE="$(du -sh "$TASK_DATASET_ROOT" 2>/dev/null | awk '{print $1}' || true)"
    echo "sampler_running poll=$TASK_POLL elapsed_seconds=$((SECONDS - TASK_WAIT_START)) en_shards=$TASK_EN_SHARDS output_size=${TASK_CURRENT_SIZE:-0}"
    pgrep -af '[p]repare_data.py sample' || true
    sleep 60
done
if pgrep -f '[p]repare_data.py sample' >/dev/null; then
    die "sampler still running after $((TASK_MAX_POLLS * 60)) seconds"
fi

echo '=== verify exact deterministic sampled datasets ==='
test -d "$TASK_TRAIN_DATA" || die "missing sampled train directory"
test -d "$TASK_DATASET_ROOT/eval" || die "missing sampled eval directory"
"$TASK_PYTHON" - "$TASK_DATASET_ROOT" <<'PY'
import sys
from pathlib import Path
from datasets import load_from_disk

root = Path(sys.argv[1])
expected = {
    "en": (35, 36_595_514, 11_822),
    "vi": (2, 927_135, 9_251),
    "zh": (2, 964_420, 9_802),
    "ru": (2, 805_963, 7_963),
    "de": (2, 974_614, 9_596),
    "ar": (2, 846_723, 8_337),
}

train_languages = {path.name for path in (root / "train").iterdir() if path.is_dir()}
eval_languages = {path.name for path in (root / "eval").iterdir() if path.is_dir()}
assert train_languages == set(expected), (train_languages, set(expected))
assert eval_languages == set(expected), (eval_languages, set(expected))

for lang, (expected_shards, expected_train_rows, expected_eval_rows) in expected.items():
    shard_paths = sorted((root / "train" / lang).glob("shard_*"))
    assert len(shard_paths) == expected_shards, (lang, len(shard_paths), expected_shards)
    train_rows = 0
    for shard_path in shard_paths:
        dataset = load_from_disk(str(shard_path))
        assert dataset.column_names == ["text"], (lang, shard_path, dataset.column_names)
        assert dataset.features["text"].dtype == "string", (lang, shard_path, dataset.features)
        assert len(dataset) > 0, (lang, shard_path)
        train_rows += len(dataset)
    evaluation = load_from_disk(str(root / "eval" / lang))
    assert evaluation.column_names == ["text"], (lang, evaluation.column_names)
    assert evaluation.features["text"].dtype == "string", (lang, evaluation.features)
    assert train_rows == expected_train_rows, (lang, train_rows, expected_train_rows)
    assert len(evaluation) == expected_eval_rows, (lang, len(evaluation), expected_eval_rows)
    print(f"{lang} shards={len(shard_paths)} train_rows={train_rows} eval_rows={len(evaluation)}")

print("EXACT_SAMPLED_DATASET_VERIFICATION_OK")
PY
du -sh "$TASK_DATASET_ROOT"

TASK_DATA_CACHE_COUNT="$(find "$TASK_DATASET_ROOT" -type f -name 'cache-*' | wc -l)"
[ "$TASK_DATA_CACHE_COUNT" -eq 0 ] \
    || die "sampled source contains $TASK_DATA_CACHE_COUNT stale cache-* files"
TASK_DATA_TMP_COUNT="$(find "$TASK_DATASET_ROOT" -mindepth 1 -name 'tmp*' | wc -l)"
[ "$TASK_DATA_TMP_COUNT" -eq 0 ] \
    || die "sampled source contains $TASK_DATA_TMP_COUNT stale tmp* paths"
echo 'sampled_source_cache_files=0 sampled_source_tmp_paths=0'

echo '=== verify local native-tied Qwen3 model ==='
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

# Recheck after the potentially long sampling wait, immediately before changing
# GPU state. A manual or duplicate launch must never be mistaken for the burn.
verify_fresh_training_paths

verify_b200_inventory() {
    mapfile -t TASK_GPU_ROWS < <(nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader)
    [ "${#TASK_GPU_ROWS[@]}" -eq 8 ] || die "expected 8 GPUs, found ${#TASK_GPU_ROWS[@]}"
    declare -gA TASK_GPU_UUIDS=()
    local row index name uuid
    for row in "${TASK_GPU_ROWS[@]}"; do
        IFS=',' read -r index name uuid <<< "$row"
        index="${index//[[:space:]]/}"
        name="${name# }"
        uuid="${uuid//[[:space:]]/}"
        [[ "$name" == *B200* ]] || die "GPU $index is not B200: $name"
        TASK_GPU_UUIDS["$uuid"]=1
        echo "gpu=$index name=$name uuid=$uuid"
    done
    [ "${#TASK_GPU_UUIDS[@]}" -eq 8 ] || die "GPU UUIDs are not unique"
}

gpu_compute_pids() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | sed '/^[[:space:]]*$/d' | tr -d ' ' | sort -nu
}

verify_b200_inventory

echo '=== validate and stop only the runner GPU-burn workers ==='
mapfile -t TASK_GPU_PROCESS_ROWS < <(
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
        --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d'
)
if [ "${#TASK_GPU_PROCESS_ROWS[@]}" -eq 0 ]; then
    echo 'gpu_burn_workers=already_absent'
else
    [ "${#TASK_GPU_PROCESS_ROWS[@]}" -eq 8 ] \
        || die "expected exactly 8 burn GPU process rows, found ${#TASK_GPU_PROCESS_ROWS[@]}"
    declare -A TASK_SEEN_PIDS=()
    declare -A TASK_SEEN_PROCESS_GPUS=()
    TASK_BURN_PIDS=()
    for TASK_ROW in "${TASK_GPU_PROCESS_ROWS[@]}"; do
        IFS=',' read -r TASK_UUID TASK_PID TASK_PROCESS_NAME TASK_USED_MEMORY <<< "$TASK_ROW"
        TASK_UUID="${TASK_UUID//[[:space:]]/}"
        TASK_PID="${TASK_PID//[[:space:]]/}"
        TASK_PROCESS_NAME="${TASK_PROCESS_NAME# }"
        TASK_USED_MEMORY="${TASK_USED_MEMORY//[[:space:]]/}"
        [[ "$TASK_PID" =~ ^[0-9]+$ ]] || die "invalid GPU PID: $TASK_PID"
        [ "$TASK_PID" -gt 1 ] || die "refusing to kill PID $TASK_PID"
        [ -n "${TASK_GPU_UUIDS[$TASK_UUID]:-}" ] || die "unknown GPU UUID: $TASK_UUID"
        [ -z "${TASK_SEEN_PIDS[$TASK_PID]:-}" ] || die "GPU PID repeated: $TASK_PID"
        [ -z "${TASK_SEEN_PROCESS_GPUS[$TASK_UUID]:-}" ] || die "multiple GPU processes on $TASK_UUID"
        [[ "$TASK_PROCESS_NAME" == *python3* ]] || die "non-Python GPU process: $TASK_PROCESS_NAME"
        [[ "$TASK_USED_MEMORY" =~ ^[0-9]+$ ]] || die "invalid GPU memory: $TASK_USED_MEMORY"
        [ "$TASK_USED_MEMORY" -ge 1000 ] && [ "$TASK_USED_MEMORY" -le 10000 ] \
            || die "unexpected burn memory for PID $TASK_PID: ${TASK_USED_MEMORY} MiB"
        [ -r "/proc/$TASK_PID/cmdline" ] || die "GPU process vanished before validation: $TASK_PID"
        TASK_CMDLINE="$(tr '\0' ' ' < "/proc/$TASK_PID/cmdline")"
        [[ "$TASK_CMDLINE" == *multiprocessing.spawn* ]] \
            && [[ "$TASK_CMDLINE" == *spawn_main* ]] \
            && [[ "$TASK_CMDLINE" == *--multiprocessing-fork* ]] \
            || die "GPU PID $TASK_PID does not match the runner burn worker: $TASK_CMDLINE"
        echo "validated_burn_pid=$TASK_PID gpu_uuid=$TASK_UUID memory_mib=$TASK_USED_MEMORY"
        TASK_SEEN_PIDS["$TASK_PID"]=1
        TASK_SEEN_PROCESS_GPUS["$TASK_UUID"]=1
        TASK_BURN_PIDS+=("$TASK_PID")
    done
    [ "${#TASK_SEEN_PROCESS_GPUS[@]}" -eq 8 ] || die "burn does not cover all 8 GPUs"
    kill -9 "${TASK_BURN_PIDS[@]}"
    echo "killed_burn_pids=${TASK_BURN_PIDS[*]}"
    sleep 5
fi

echo '=== verify all GPUs free after stopping burn ==='
mapfile -t TASK_REMAINING_PIDS < <(gpu_compute_pids)
[ "${#TASK_REMAINING_PIDS[@]}" -eq 0 ] \
    || die "GPU compute processes remain: ${TASK_REMAINING_PIDS[*]}"
nvidia-smi
echo 'ALL_8_B200_GPUS_FREE_AFTER_BURN_STOP'

echo '=== activate and validate sparse_emb conda environment ==='
eval "$("$TASK_CONDA" shell.bash hook)"
conda activate sparse_emb
[ "${CONDA_DEFAULT_ENV:-}" = sparse_emb ] || die "failed to activate sparse_emb"
[ "$(command -v python3.11)" = "$TASK_PYTHON" ] \
    || die "wrong Python after activation: $(command -v python3.11)"
"$TASK_PYTHON" -c 'import accelerate, datasets, torch, transformers; print("TRAINING_ENV_OK", torch.__version__, transformers.__version__, datasets.__version__, accelerate.__version__)'

echo '=== install and verify Accelerate configuration ==='
mkdir -p "$(dirname "$TASK_ACCELERATE_TARGET")"
cp "$TASK_ACCELERATE_SOURCE" "$TASK_ACCELERATE_TARGET"
cmp "$TASK_ACCELERATE_SOURCE" "$TASK_ACCELERATE_TARGET"
grep -Fxq 'distributed_type: MULTI_GPU' "$TASK_ACCELERATE_TARGET"
grep -Fxq 'mixed_precision: bf16' "$TASK_ACCELERATE_TARGET"
grep -Fxq 'num_processes: 8' "$TASK_ACCELERATE_TARGET"
echo "accelerate_config=$TASK_ACCELERATE_TARGET"

echo '=== wait 60 seconds, then verify GPUs are still free ==='
sleep 60
verify_b200_inventory
mapfile -t TASK_FINAL_GPU_PIDS < <(gpu_compute_pids)
[ "${#TASK_FINAL_GPU_PIDS[@]}" -eq 0 ] \
    || die "GPU compute processes appeared during the 60-second guard: ${TASK_FINAL_GPU_PIDS[*]}"
nvidia-smi
echo 'ALL_8_B200_GPUS_STILL_FREE_AFTER_60_SECONDS'

mkdir -p "$TASK_OUTPUT_BASE"
export SPARSE_EMB_PYTHON="$TASK_PYTHON"
export SPARSE_EMB_MODEL_DIR="$TASK_MODEL_DIR"
export SPARSE_EMB_DATA_DIR="$TASK_TRAIN_DATA"
export SPARSE_EMB_OUTPUT_BASE="$TASK_OUTPUT_BASE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo '=== resolve the dense tied baseline by exact experiment name ==='
TASK_EXPERIMENT_INDEX="$("$TASK_PYTHON" - "$TASK_OUTPUT_DIR" <<'PY'
import sys
from run_experiments import EXPERIMENT_COMMANDS

matches = [(index, experiment) for index, experiment in enumerate(EXPERIMENT_COMMANDS)
           if experiment.get("name") == "dense_tied_baseline_b200"]
assert len(matches) == 1, matches
index, experiment = matches[0]
assert experiment["cmd"] == "bash scripts/train_dense_tied_baseline_b200.sh", experiment
assert experiment["output_dir"] == sys.argv[1], (experiment["output_dir"], sys.argv[1])
assert experiment.get("require_fresh_output") is True, experiment
print(index)
PY
)"
[[ "$TASK_EXPERIMENT_INDEX" =~ ^[0-9]+$ ]] \
    || die "could not resolve baseline experiment index: $TASK_EXPERIMENT_INDEX"

echo "experiment_index=$TASK_EXPERIMENT_INDEX"
echo 'architecture=native_Qwen3_dense_input_output_exactly_tied'
echo 'precision=bf16 seed=42 block_size=2048'
echo 'batch=16_per_device_x_4_accumulation_x_8_gpus=512_sequences_per_update'
echo 'schedule=original_one_epoch_with_runner_stop_at_checkpoint_10000'
echo "output_dir=$TASK_OUTPUT_DIR"
echo "log_dir=$TASK_LOG_DIR"
echo '=== launch dense_tied_baseline_b200 ==='
exec "$TASK_PYTHON" -u run_experiments.py \
    --experiments "$TASK_EXPERIMENT_INDEX" \
    --stop-at-step 10000 \
    --log-dir "$TASK_LOG_DIR"
