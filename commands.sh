#1 +120+a
#th2-fresh-rerun-b200-global-lr-tied-r128-then-dense-tied-10k-20260824
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
TASK_HF_DATASETS_CACHE=/mnt/local/.cache/huggingface/datasets
TASK_ACCELERATE_SOURCE="$TASK_PROJECT_DIR/resources/accelerate_config.yaml"
TASK_ACCELERATE_DEST=/mnt/local/.cache/huggingface/accelerate/default_config.yaml

echo '=== activate and validate sparse_emb ==='
date -u
hostname
test -x "$TASK_CONDA"
eval "$("$TASK_CONDA" shell.bash hook)"
conda activate sparse_emb
test "$CONDA_DEFAULT_ENV" = sparse_emb
test "$(command -v python3.11)" = "$TASK_PYTHON"
"$TASK_PYTHON" -c 'import accelerate, datasets, torch, transformers; print("training_imports=OK", torch.__version__, accelerate.__version__, transformers.__version__, datasets.__version__)'

echo '=== verify all eight B200 GPUs are completely free ==='
mapfile -t TASK_GPU_NAMES < <(
    nvidia-smi --query-gpu=name --format=csv,noheader \
        | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
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
test "${#TASK_GPU_PIDS[@]}" -eq 0 || {
    echo "ERROR: GPUs have compute processes: ${TASK_GPU_PIDS[*]}"
    exit 1
}
echo 'TH2 ALL 8 B200 GPUS FREE BEFORE FRESH RERUN'

echo '=== verify previous matched run left no process, output, log, or cache ==='
if pgrep -af 'run_experiments.py.*--experiments 13 14|global_lowrank_tied_r128_b200|dense_tied_baseline_b200'; then
    echo 'ERROR: previous matched-control process remains'
    exit 1
fi
for path in "$TASK_GLOBAL_OUTPUT" "$TASK_DENSE_OUTPUT" "$TASK_LOG_DIR" \
            "$TASK_HF_DATASETS_CACHE"; do
    test ! -e "$path" || {
        echo "ERROR: stale path exists: $path"
        du -sh "$path" || true
        exit 1
    }
    echo "absent=$path"
done
TASK_DATA_CACHE_COUNT=$(find "$TASK_DATA_DIR" -type f -name 'cache-*' | wc -l)
test "$TASK_DATA_CACHE_COUNT" -eq 0
echo 'dataset_preprocessing_cache_files=0'

echo '=== remove and verify scoped dataset tmp-* leftovers ==='
mapfile -d '' TASK_DATA_TMP_PATHS < <(
    find "$TASK_DATA_DIR" -mindepth 1 -name 'tmp*' -print0
)
echo "dataset_tmp_paths_before=${#TASK_DATA_TMP_PATHS[@]}"
for path in "${TASK_DATA_TMP_PATHS[@]}"; do
    case "$path" in
        "$TASK_DATA_DIR"/*) rm -rf -- "$path" ;;
        *) echo "REFUSE unexpected temporary path: $path"; exit 1 ;;
    esac
done
test "$(find "$TASK_DATA_DIR" -mindepth 1 -name 'tmp*' | wc -l)" -eq 0
echo 'dataset_tmp_paths_after=0'
if [ -d "$TASK_PROJECT_DIR/wandb" ]; then
    test "$(find "$TASK_PROJECT_DIR/wandb" -mindepth 1 -maxdepth 1 -type d \
        -name 'offline-run-20260824_1942*' | wc -l)" -eq 0
fi
echo 'TH2 PREVIOUS MATCHED-CONTROL OUTPUTS AND CACHES ABSENT'

echo '=== verify local model and untouched sampled source data ==='
test -s "$TASK_MODEL_DIR/config.json"
test -s "$TASK_MODEL_DIR/tokenizer.json"
for lang in en vi zh ru de ar; do
    test -d "$TASK_DATA_DIR/$lang"
    TASK_ARROW_COUNT=$(find "$TASK_DATA_DIR/$lang" -type f -name '*.arrow' \
        ! -name 'cache-*' | wc -l)
    test "$TASK_ARROW_COUNT" -gt 0
    echo "source_language=$lang arrow_files=$TASK_ARROW_COUNT"
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

echo '=== install and verify exact eight-GPU bf16 Accelerate configuration ==='
test -s "$TASK_ACCELERATE_SOURCE"
mkdir -p "$(dirname "$TASK_ACCELERATE_DEST")"
cp "$TASK_ACCELERATE_SOURCE" "$TASK_ACCELERATE_DEST"
cmp "$TASK_ACCELERATE_SOURCE" "$TASK_ACCELERATE_DEST"
grep -Fx 'distributed_type: MULTI_GPU' "$TASK_ACCELERATE_DEST"
grep -Fx 'mixed_precision: bf16' "$TASK_ACCELERATE_DEST"
grep -Fx 'num_processes: 8' "$TASK_ACCELERATE_DEST"
echo "accelerate_config=$TASK_ACCELERATE_DEST"

echo '=== verify matched sequential experiment definitions ==='
export SPARSE_EMB_PYTHON="$TASK_PYTHON"
export SPARSE_EMB_MODEL_DIR="$TASK_MODEL_DIR"
export SPARSE_EMB_DATA_DIR="$TASK_DATA_DIR"
export SPARSE_EMB_OUTPUT_BASE="$TASK_OUTPUT_BASE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
"$TASK_PYTHON" -u run_experiments.py --list \
    | grep -F '[13] global_lowrank_tied_r128_b200:'
"$TASK_PYTHON" -u run_experiments.py --list \
    | grep -F '[14] dense_tied_baseline_b200:'
echo 'experiment_order=global_lowrank_tied_r128_b200,dense_tied_baseline_b200'
echo 'global_architecture=rank128_input_output_same_parameters_for_entire_training'
echo 'dense_architecture=native_dense_input_output_same_parameters_for_entire_training'
echo 'batch_configuration=16_per_device_x_4_accum_x_8_gpus=512_sequences_per_step'
echo 'precision=bf16 stop_checkpoint=10000 original_one_epoch_schedule=true'

echo '=== launch fresh global LR tied, then dense tied, each to checkpoint 10000 ==='
exec "$TASK_PYTHON" -u run_experiments.py \
    --experiments 13 14 \
    --stop-at-step 10000 \
    --log-dir "$TASK_LOG_DIR"
