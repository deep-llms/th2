#1 +120+a
#th2-launch-b200-matched-global-lr-then-dense-tied-10k-20260824-retry1
set -euo pipefail

TASK_PROJECT_DIR="$PWD"
TASK_CONDA=/mnt/local/conda-py311/bin/conda
TASK_PYTHON=/mnt/local/conda-py311/envs/sparse_emb/bin/python3.11
TASK_MODEL_DIR=/mnt/local/_models/@PROJECT@/Qwen3-0.6B
TASK_DATA_DIR=/mnt/local/_data/@PROJECT@/data/Qwen_Qwen3-0.6B/train
TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_GLOBAL_OUTPUT="$TASK_OUTPUT_BASE/global_lowrank_tied_r128_b200"
TASK_DENSE_OUTPUT="$TASK_OUTPUT_BASE/dense_tied_baseline_b200"
TASK_LOG_DIR="$TASK_OUTPUT_BASE/logs/b200_matched_global_lr_then_dense_tied_20260824"
TASK_ACCELERATE_SOURCE="$TASK_PROJECT_DIR/resources/accelerate_config.yaml"
TASK_ACCELERATE_DEST="$HOME/.cache/huggingface/accelerate/default_config.yaml"

echo '=== activate and validate sparse_emb ==='
date -u
hostname
test -x "$TASK_CONDA"
eval "$("$TASK_CONDA" shell.bash hook)"
conda activate sparse_emb
test "$CONDA_DEFAULT_ENV" = sparse_emb
test "$(command -v python3.11)" = "$TASK_PYTHON"
"$TASK_PYTHON" -c 'import accelerate, datasets, torch, transformers; print("training_imports=OK", torch.__version__, accelerate.__version__, transformers.__version__, datasets.__version__)'

echo '=== verify all eight B200 GPUs are free ==='
mapfile -t TASK_GPU_NAMES < <(
    nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
)
test "${#TASK_GPU_NAMES[@]}" -eq 8
for index in "${!TASK_GPU_NAMES[@]}"; do
    echo "gpu=$index name=${TASK_GPU_NAMES[$index]}"
    case "${TASK_GPU_NAMES[$index]}" in
        *B200*) ;;
        *) echo "ERROR: GPU $index is not a B200"; exit 1 ;;
    esac
done
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
mapfile -t TASK_GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -u
)
if [ "${#TASK_GPU_PIDS[@]}" -ne 0 ]; then
    echo "ERROR: GPUs are not free; compute PIDs: ${TASK_GPU_PIDS[*]}"
    for pid in "${TASK_GPU_PIDS[@]}"; do
        tr '\0' ' ' < "/proc/$pid/cmdline" || true
        echo
    done
    exit 1
fi
echo 'TH2 ALL 8 B200 GPUS FREE'

echo '=== verify local model, sampled data, and native dense tying ==='
test -s "$TASK_MODEL_DIR/config.json"
test -s "$TASK_MODEL_DIR/tokenizer.json"
for lang in en vi zh ru de ar; do
    test -d "$TASK_DATA_DIR/$lang"
    test "$(find "$TASK_DATA_DIR/$lang" -type f -name '*.arrow' | wc -l)" -gt 0
done
"$TASK_PYTHON" - "$TASK_MODEL_DIR/config.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    config = json.load(handle)
assert config.get("model_type") == "qwen3", config.get("model_type")
assert config.get("tie_word_embeddings") is True, config.get("tie_word_embeddings")
assert config.get("hidden_size") == 1024, config.get("hidden_size")
assert config.get("vocab_size") == 151936, config.get("vocab_size")
print("model_config=Qwen3-0.6B dense_native_tying=true")
PY

echo '=== require fresh, scoped output and log directories ==='
for output in "$TASK_GLOBAL_OUTPUT" "$TASK_DENSE_OUTPUT"; do
    case "$output" in
        /mnt/local/_outputs/*/global_lowrank_tied_r128_b200|/mnt/local/_outputs/*/dense_tied_baseline_b200) ;;
        *) echo "REFUSE: unexpected output path: $output"; exit 1 ;;
    esac
    if [ -e "$output" ]; then
        echo "REFUSE: experiment output already exists: $output"
        du -sh "$output" || true
        exit 1
    fi
done
if [ -e "$TASK_LOG_DIR" ]; then
    echo "REFUSE: experiment log directory already exists: $TASK_LOG_DIR"
    exit 1
fi

echo '=== install and verify Accelerate configuration ==='
test -s "$TASK_ACCELERATE_SOURCE"
mkdir -p "$(dirname "$TASK_ACCELERATE_DEST")"
cp "$TASK_ACCELERATE_SOURCE" "$TASK_ACCELERATE_DEST"
cmp "$TASK_ACCELERATE_SOURCE" "$TASK_ACCELERATE_DEST"
grep -Fx 'distributed_type: MULTI_GPU' "$TASK_ACCELERATE_DEST"
grep -Fx 'mixed_precision: bf16' "$TASK_ACCELERATE_DEST"
grep -Fx 'num_processes: 8' "$TASK_ACCELERATE_DEST"
echo "accelerate_config=$TASK_ACCELERATE_DEST"

echo '=== matched sequential experiment preflight ==='
export SPARSE_EMB_PYTHON="$TASK_PYTHON"
export SPARSE_EMB_MODEL_DIR="$TASK_MODEL_DIR"
export SPARSE_EMB_DATA_DIR="$TASK_DATA_DIR"
export SPARSE_EMB_OUTPUT_BASE="$TASK_OUTPUT_BASE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
"$TASK_PYTHON" -u run_experiments.py --list | grep -F '[13] global_lowrank_tied_r128_b200:'
"$TASK_PYTHON" -u run_experiments.py --list | grep -F '[14] dense_tied_baseline_b200:'
echo 'experiment_order=global_lowrank_tied_r128_b200,dense_tied_baseline_b200'
echo 'global_architecture=rank128_input_output_same_parameters_for_entire_training'
echo 'dense_architecture=native_dense_input_output_same_parameters_for_entire_training'
echo 'batch_configuration=16_per_device_x_4_accum_x_8_gpus=512_sequences_per_step'
echo 'precision=bf16 stop_checkpoint=10000'

echo '=== launch global LR tied, then dense tied, each to checkpoint 10000 ==='
exec "$TASK_PYTHON" -u run_experiments.py \
    --experiments 13 14 \
    --stop-at-step 10000 \
    --log-dir "$TASK_LOG_DIR"
