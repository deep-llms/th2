#1 +120+a
#th2-stop-reclean-relaunch-b200-matched-tied-controls-20260824
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

is_matched_control_process() {
    local cmd=$1
    [[ "$cmd" == *python* ]] && {
        [[ "$cmd" == *"run_experiments.py --experiments 13 14"* ]] \
            || [[ "$cmd" == *"global_lowrank_tied_r128_b200"* ]] \
            || [[ "$cmd" == *"dense_tied_baseline_b200"* ]]
    }
}

matched_control_pids() {
    local proc pid cmd
    for proc in /proc/[0-9]*; do
        pid=${proc#/proc/}
        [ -r "$proc/cmdline" ] || continue
        cmd=$(tr '\0' ' ' < "$proc/cmdline" 2>/dev/null || true)
        if is_matched_control_process "$cmd"; then
            printf '%s\n' "$pid"
        fi
    done | sort -nu
}

echo '=== confirm and stop only the currently running matched controls ==='
date -u
hostname
mapfile -t TASK_PIDS < <(matched_control_pids)
test "${#TASK_PIDS[@]}" -gt 0 || {
    echo 'ERROR: expected prior matched-control run is not running'
    exit 1
}
echo "matched_control_processes=${#TASK_PIDS[@]}"
for pid in "${TASK_PIDS[@]}"; do
    printf 'pid=%s cmd=' "$pid"
    tr '\0' ' ' < "/proc/$pid/cmdline" || true
    echo
done
kill -TERM "${TASK_PIDS[@]}" 2>/dev/null || true
for attempt in 1 2 3; do
    sleep 5
    mapfile -t TASK_PIDS < <(matched_control_pids)
    [ "${#TASK_PIDS[@]}" -eq 0 ] && break
    echo "remaining_after_term_attempt_${attempt}=${TASK_PIDS[*]}"
    kill -TERM "${TASK_PIDS[@]}" 2>/dev/null || true
done
mapfile -t TASK_PIDS < <(matched_control_pids)
if [ "${#TASK_PIDS[@]}" -gt 0 ]; then
    echo "force_kill=${TASK_PIDS[*]}"
    kill -KILL "${TASK_PIDS[@]}" 2>/dev/null || true
    sleep 5
fi
mapfile -t TASK_PIDS < <(matched_control_pids)
test "${#TASK_PIDS[@]}" -eq 0
echo 'matched_control_processes_after_stop=0'

echo '=== verify all eight B200 GPUs are free before deleting anything ==='
mapfile -t TASK_GPU_NAMES < <(
    nvidia-smi --query-gpu=name --format=csv,noheader \
        | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
)
test "${#TASK_GPU_NAMES[@]}" -eq 8
for index in "${!TASK_GPU_NAMES[@]}"; do
    case "${TASK_GPU_NAMES[$index]}" in
        *B200*) ;;
        *) echo "ERROR: GPU $index is not a B200"; exit 1 ;;
    esac
done
mapfile -t TASK_GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -u
)
test "${#TASK_GPU_PIDS[@]}" -eq 0 || {
    echo "ERROR: unexpected GPU processes remain: ${TASK_GPU_PIDS[*]}"
    exit 1
}
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
echo 'TH2 ALL 8 B200 GPUS FREE AFTER STOP'

echo '=== inventory exact outputs and caches before scoped cleanup ==='
for path in "$TASK_GLOBAL_OUTPUT" "$TASK_DENSE_OUTPUT" "$TASK_LOG_DIR" \
            "$TASK_HF_DATASETS_CACHE"; do
    if [ -e "$path" ]; then
        du -sh "$path"
    else
        echo "absent=$path"
    fi
done
mapfile -d '' TASK_DATA_CACHE_PATHS < <(
    find "$TASK_DATA_DIR" -mindepth 1 \
        \( -name 'cache-*' -o -name 'tmp*' \) -print0
)
echo "dataset_cache_or_tmp_paths_before=${#TASK_DATA_CACHE_PATHS[@]}"

echo '=== remove only current matched-control outputs and caches ==='
rm -rf -- "$TASK_GLOBAL_OUTPUT"
rm -rf -- "$TASK_DENSE_OUTPUT"
rm -rf -- "$TASK_LOG_DIR"
rm -rf -- "$TASK_HF_DATASETS_CACHE"
for path in "${TASK_DATA_CACHE_PATHS[@]}"; do
    case "$path" in
        "$TASK_DATA_DIR"/*) rm -rf -- "$path" ;;
        *) echo "REFUSE unexpected cache path: $path"; exit 1 ;;
    esac
done

echo '=== remove only W&B runs belonging to these matched outputs ==='
if [ -d "$TASK_PROJECT_DIR/wandb" ]; then
    mapfile -d '' TASK_WANDB_METADATA < <(
        find "$TASK_PROJECT_DIR/wandb" -mindepth 3 -maxdepth 3 -type f \
            -path '*/files/wandb-metadata.json' -print0
    )
    for metadata in "${TASK_WANDB_METADATA[@]}"; do
        if grep -Fq 'global_lowrank_tied_r128_b200' "$metadata" \
                || grep -Fq 'dense_tied_baseline_b200' "$metadata"; then
            run_dir=$(dirname "$(dirname "$metadata")")
            case "$run_dir" in
                "$TASK_PROJECT_DIR"/wandb/offline-run-*) rm -rf -- "$run_dir" ;;
                *) echo "REFUSE unexpected W&B path: $run_dir"; exit 1 ;;
            esac
        fi
    done
fi

echo '=== verify clean state and preserve raw sampled data ==='
for path in "$TASK_GLOBAL_OUTPUT" "$TASK_DENSE_OUTPUT" "$TASK_LOG_DIR" \
            "$TASK_HF_DATASETS_CACHE"; do
    test ! -e "$path"
    echo "absent=$path"
done
test "$(find "$TASK_DATA_DIR" -mindepth 1 \
    \( -name 'cache-*' -o -name 'tmp*' \) | wc -l)" -eq 0
echo 'dataset_cache_or_tmp_paths_after=0'
for lang in en vi zh ru de ar; do
    test -d "$TASK_DATA_DIR/$lang"
    TASK_ARROW_COUNT=$(find "$TASK_DATA_DIR/$lang" -type f -name '*.arrow' \
        ! -name 'cache-*' | wc -l)
    test "$TASK_ARROW_COUNT" -gt 0
    echo "source_language=$lang arrow_files=$TASK_ARROW_COUNT"
done
echo 'TH2 MATCHED-CONTROL OUTPUTS CACHES AND TMP PATHS CLEAN'

echo '=== activate and validate sparse_emb ==='
test -x "$TASK_CONDA"
eval "$("$TASK_CONDA" shell.bash hook)"
conda activate sparse_emb
test "$CONDA_DEFAULT_ENV" = sparse_emb
test "$(command -v python3.11)" = "$TASK_PYTHON"
"$TASK_PYTHON" -c 'import accelerate, datasets, torch, transformers; print("training_imports=OK", torch.__version__, accelerate.__version__, transformers.__version__, datasets.__version__)'

echo '=== verify local model and native dense tying ==='
test -s "$TASK_MODEL_DIR/config.json"
test -s "$TASK_MODEL_DIR/tokenizer.json"
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

echo '=== copy and verify exact eight-GPU bf16 Accelerate configuration ==='
test -s "$TASK_ACCELERATE_SOURCE"
mkdir -p "$(dirname "$TASK_ACCELERATE_DEST")"
cp "$TASK_ACCELERATE_SOURCE" "$TASK_ACCELERATE_DEST"
cmp "$TASK_ACCELERATE_SOURCE" "$TASK_ACCELERATE_DEST"
grep -Fx 'distributed_type: MULTI_GPU' "$TASK_ACCELERATE_DEST"
grep -Fx 'mixed_precision: bf16' "$TASK_ACCELERATE_DEST"
grep -Fx 'num_processes: 8' "$TASK_ACCELERATE_DEST"
echo "accelerate_config=$TASK_ACCELERATE_DEST"

echo '=== wait one minute after Accelerate copy ==='
sleep 60

echo '=== repeat clean-state and GPU checks immediately before launch ==='
mapfile -t TASK_PIDS < <(matched_control_pids)
test "${#TASK_PIDS[@]}" -eq 0
for path in "$TASK_GLOBAL_OUTPUT" "$TASK_DENSE_OUTPUT" "$TASK_LOG_DIR" \
            "$TASK_HF_DATASETS_CACHE"; do
    test ! -e "$path"
done
test "$(find "$TASK_DATA_DIR" -mindepth 1 \
    \( -name 'cache-*' -o -name 'tmp*' \) | wc -l)" -eq 0
mapfile -t TASK_GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -u
)
test "${#TASK_GPU_PIDS[@]}" -eq 0
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
echo 'TH2 ALL 8 B200 GPUS STILL FREE AFTER ONE-MINUTE WAIT'

echo '=== fresh matched launch ==='
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
echo 'batch_configuration=16_per_device_x_4_accum_x_8_gpus=512_sequences_per_step'
echo 'precision=bf16 stop_checkpoint=10000 original_one_epoch_schedule=true'
echo 'TH2 FRESH MATCHED-CONTROL RELAUNCH PREFLIGHT COMPLETE'
exec "$TASK_PYTHON" -u run_experiments.py \
    --experiments 13 14 \
    --stop-at-step 10000 \
    --log-dir "$TASK_LOG_DIR"
