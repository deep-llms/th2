#1 +180+a
#th2-train-nested-ladder-and-matched-groupreduce-10k-20260829-a01
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

TASK_PROJECT_DIR=/mnt/local/@PROJECT@
TASK_CONDA=/mnt/local/conda-py311/bin/conda
TASK_PYTHON=/mnt/local/conda-py311/envs/sparse_emb/bin/python3.11
TASK_MODEL_DIR=/mnt/local/_models/@PROJECT@/Qwen3-0.6B
TASK_DATA_DIR=/mnt/local/_data/@PROJECT@/data/Qwen_Qwen3-0.6B/train
TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_LOG_DIR="$TASK_OUTPUT_BASE/logs/nested_ladder_phase1_20260829"
TASK_NESTED_OUTPUT="$TASK_OUTPUT_BASE/nested_ladder_tied_t4"
TASK_GROUPREDUCE_OUTPUT="$TASK_OUTPUT_BASE/groupreduce_matched_nested_tied_t4"
TASK_ACCELERATE_SOURCE="$TASK_PROJECT_DIR/resources/accelerate_config.yaml"
TASK_ACCELERATE_TARGET=/mnt/local/.cache/huggingface/accelerate/default_config.yaml

echo '=== identity and static preflight ==='
date -u
hostname
cd "$TASK_PROJECT_DIR"
test -x "$TASK_CONDA"
test -x "$TASK_PYTHON"
test -s scripts/train_nested_ladder_tied.sh
test -s scripts/train_groupreduce_matched_nested_tied.sh
for TASK_LAUNCHER in \
    scripts/train_nested_ladder_tied.sh \
    scripts/train_groupreduce_matched_nested_tied.sh; do
    grep -Eq '^[[:space:]]*--ddp_find_unused_parameters[[:space:]]+false([[:space:]]|$)' \
        "$TASK_LAUNCHER" || die "missing DDP false setting in $TASK_LAUNCHER"
    grep -Eq '^[[:space:]]*--bf16([[:space:]]|$)' "$TASK_LAUNCHER" \
        || die "missing BF16 setting in $TASK_LAUNCHER"
    if grep -Eq '^[[:space:]]*--max_steps([[:space:]]|$)' "$TASK_LAUNCHER"; then
        die "launcher must retain the one-epoch schedule: $TASK_LAUNCHER"
    fi
done
test -s resources/token_freq_sample10.npz
echo 'c2694ea0b34f69119472a4262013b2387166aaa1981c80ac5b30c32b72c5741c  resources/token_freq_sample10.npz' \
    | sha256sum -c -
test -s "$TASK_ACCELERATE_SOURCE"
test -s "$TASK_MODEL_DIR/config.json"
test -s "$TASK_MODEL_DIR/tokenizer.json"
test -d "$TASK_DATA_DIR"
[[ ! -e "$TASK_NESTED_OUTPUT" ]] \
    || die "fresh Nested Ladder output already exists: $TASK_NESTED_OUTPUT"
[[ ! -e "$TASK_GROUPREDUCE_OUTPUT" ]] \
    || die "fresh matched GroupReduce output already exists: $TASK_GROUPREDUCE_OUTPUT"
[[ ! -e "$TASK_LOG_DIR" ]] \
    || die "fresh experiment log directory already exists: $TASK_LOG_DIR"
for TASK_LANG in en vi zh ru de ar; do
    test -d "$TASK_DATA_DIR/$TASK_LANG" \
        || die "missing training language: $TASK_LANG"
    find "$TASK_DATA_DIR/$TASK_LANG" -type f -name '*.arrow' \
        ! -name 'cache-*' -print -quit | grep -q . \
        || die "no source Arrow file for $TASK_LANG"
done

echo '=== activate and verify sparse_emb ==='
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
print(
    "TRAINING_ENV_OK",
    f"torch={torch.__version__}",
    f"transformers={transformers.__version__}",
    f"datasets={datasets.__version__}",
    f"accelerate={accelerate.__version__}",
)
PY
"$TASK_PYTHON" - "$TASK_MODEL_DIR/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)
assert config.get("model_type") == "qwen3", config.get("model_type")
assert config.get("tie_word_embeddings") is True, config.get("tie_word_embeddings")
assert config.get("hidden_size") == 1024, config.get("hidden_size")
assert config.get("vocab_size") == 151936, config.get("vocab_size")
print("QWEN3_0.6B_NATIVE_CONFIG_OK")
PY

echo '=== verify current GPU processes are only the known project burns ==='
validate_b200_node
if pgrep -af '[t]rain.py|[r]un_experiments.py|[a]ccelerate.commands.launch|[a]ccelerate launch'; then
    die 'training or Accelerate process exists before burn reclamation'
fi
mapfile -t TASK_BURN_PIDS < <(gpu_pids)
[[ "${#TASK_BURN_PIDS[@]}" -eq 8 ]] \
    || die "expected 8 supervised burn PIDs, found ${#TASK_BURN_PIDS[@]}"
for TASK_PID in "${TASK_BURN_PIDS[@]}"; do
    TASK_CMDLINE="$(tr '\0' ' ' < "/proc/$TASK_PID/cmdline")"
    echo "verified_burn_pid=$TASK_PID cmdline=$TASK_CMDLINE"
    [[ "$TASK_CMDLINE" == *scripts/gpu_burn.py* ]] \
        || die "unexpected GPU process before kill: $TASK_PID $TASK_CMDLINE"
done

echo '=== kill verified GPU-burn workers only ==='
kill -9 "${TASK_BURN_PIDS[@]}"

echo '=== sleep 60 seconds after burn kill ==='
sleep 60

echo '=== first free-GPU check ==='
require_free_gpus '60 SECONDS AFTER BURN KILL'

echo '=== copy and verify exact eight-GPU bf16 Accelerate config ==='
mkdir -p "$(dirname "$TASK_ACCELERATE_TARGET")"
cp "$TASK_ACCELERATE_SOURCE" "$TASK_ACCELERATE_TARGET"
cmp "$TASK_ACCELERATE_SOURCE" "$TASK_ACCELERATE_TARGET"
grep -Fxq 'distributed_type: MULTI_GPU' "$TASK_ACCELERATE_TARGET"
grep -Fxq 'mixed_precision: bf16' "$TASK_ACCELERATE_TARGET"
grep -Fxq 'num_processes: 8' "$TASK_ACCELERATE_TARGET"

echo '=== sleep 60 seconds after Accelerate config copy ==='
sleep 60

echo '=== second free-GPU check ==='
require_free_gpus '60 SECONDS AFTER ACCELERATE CONFIG COPY'

export SPARSE_EMB_PYTHON="$TASK_PYTHON"
export SPARSE_EMB_MODEL_DIR="$TASK_MODEL_DIR"
export SPARSE_EMB_DATA_DIR="$TASK_DATA_DIR"
export SPARSE_EMB_OUTPUT_BASE="$TASK_OUTPUT_BASE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo '=== resolve exact experiment definitions by name ==='
mapfile -t TASK_EXPERIMENT_INDICES < <(
    "$TASK_PYTHON" - <<'PY'
from run_experiments import EXPERIMENT_COMMANDS

expected = (
    ("nested_ladder_tied_t4", "bash scripts/train_nested_ladder_tied.sh"),
    (
        "groupreduce_matched_nested_tied_t4",
        "bash scripts/train_groupreduce_matched_nested_tied.sh",
    ),
)
for name, command in expected:
    matches = [
        (index, experiment)
        for index, experiment in enumerate(EXPERIMENT_COMMANDS)
        if experiment.get("name") == name
    ]
    assert len(matches) == 1, (name, matches)
    index, experiment = matches[0]
    assert experiment["cmd"] == command, experiment
    assert experiment.get("require_fresh_output") is True, experiment
    print(index)
PY
)
[[ "${#TASK_EXPERIMENT_INDICES[@]}" -eq 2 ]] \
    || die 'failed to resolve exactly two experiments'
printf 'selected_experiment_indices=%s %s\n' \
    "${TASK_EXPERIMENT_INDICES[0]}" "${TASK_EXPERIMENT_INDICES[1]}"
echo 'experiment_order=nested_ladder_tied_t4 then groupreduce_matched_nested_tied_t4'
echo 'precision=bf16 seed=42 block_size=2048'
echo 'batch=16_per_device_x_4_accumulation_x_8_gpus=512_sequences_per_update'
echo 'schedule=one_epoch_with_runner_stop_at_checkpoint_10000'
echo 'ddp_find_unused_parameters=false for both tied custom architectures'

echo '=== start sequential Phase-1 training ==='
"$TASK_PYTHON" -u run_experiments.py \
    --experiments "${TASK_EXPERIMENT_INDICES[@]}" \
    --stop-at-step 10000 \
    --log-dir "$TASK_LOG_DIR"
