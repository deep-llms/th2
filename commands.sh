#1 +120+a
#th2-watch-sample-verify-stop-burn-launch-independent-lr-20260823
set -euo pipefail

echo '=== supervise sampling through independent-LR training launch ==='
date -u
hostname

TASK_PROJECT_DIR="$PWD"
TASK_CONDA="/mnt/local/conda-py311/bin/conda"
TASK_PYTHON="/mnt/local/conda-py311/envs/sparse_emb/bin/python3.11"
TASK_DATA_ROOT="/mnt/local/_data/@PROJECT@/data"
TASK_DATASET_ROOT="$TASK_DATA_ROOT/Qwen_Qwen3-0.6B"
TASK_TRAIN_DATA="$TASK_DATASET_ROOT/train"
TASK_MODEL_DIR="/mnt/local/_models/@PROJECT@/Qwen3-0.6B"
TASK_OUTPUT_BASE="/mnt/local/_outputs/@PROJECT@"
TASK_OUTPUT_DIR="$TASK_OUTPUT_BASE/lowrank_independent_output_r128"
TASK_LOG_DIR="$TASK_OUTPUT_BASE/logs/lowrank_independent_output_r128"
TASK_ACCELERATE_SOURCE="$TASK_PROJECT_DIR/resources/accelerate_config.yaml"
TASK_ACCELERATE_TARGET="$HOME/.cache/huggingface/accelerate/default_config.yaml"

echo '=== wait for the existing sampler ==='
TASK_WAIT_START="$SECONDS"
while pgrep -f '[p]repare_data.py sample' >/dev/null; do
    TASK_ELAPSED="$((SECONDS - TASK_WAIT_START))"
    TASK_EN_SHARDS="$(find "$TASK_TRAIN_DATA/en" -mindepth 1 -maxdepth 1 -type d -name 'shard_*' 2>/dev/null | wc -l || true)"
    TASK_CURRENT_SIZE="$(du -sh "$TASK_DATASET_ROOT" 2>/dev/null | awk '{print $1}' || true)"
    echo "sampler_running elapsed_seconds=$TASK_ELAPSED en_shards=$TASK_EN_SHARDS output_size=${TASK_CURRENT_SIZE:-0}"
    pgrep -af '[p]repare_data.py sample'
    sleep 60
done
echo "sampler_process_gone wait_seconds=$((SECONDS - TASK_WAIT_START))"

echo '=== activate and validate sparse_emb ==='
test -x "$TASK_CONDA"
eval "$("$TASK_CONDA" shell.bash hook)"
conda activate sparse_emb
test "$CONDA_DEFAULT_ENV" = sparse_emb
test -x "$TASK_PYTHON"
test "$(command -v python3.11)" = "$TASK_PYTHON"
"$TASK_PYTHON" -c 'import accelerate, datasets, pyarrow, torch, transformers; print("training_imports=OK", torch.__version__, transformers.__version__, datasets.__version__, accelerate.__version__)'

echo '=== verify exact H100-compatible sampled datasets ==='
test -d "$TASK_TRAIN_DATA"
test -d "$TASK_DATASET_ROOT/eval"
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

actual_languages = {path.name for path in (root / "train").iterdir() if path.is_dir()}
assert actual_languages == set(expected), (actual_languages, set(expected))
actual_eval_languages = {path.name for path in (root / "eval").iterdir() if path.is_dir()}
assert actual_eval_languages == set(expected), (actual_eval_languages, set(expected))

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
print("H100_EXACT_DATASET_COUNTS_OK")
PY
du -sh "$TASK_DATASET_ROOT"
df -h "$TASK_DATA_ROOT"
echo 'TH2 SAMPLED DATA VERIFICATION OK'

echo '=== identify and stop only GPU burn workers ==='
TASK_GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
echo "gpu_count=$TASK_GPU_COUNT"
test "$TASK_GPU_COUNT" -eq 8
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader

TASK_GPU_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d' | sort -nu)"
if [ -z "$TASK_GPU_PIDS" ]; then
    echo 'gpu_burn_workers=already_absent'
else
    for TASK_PID in $TASK_GPU_PIDS; do
        TASK_CMDLINE="$(tr '\0' ' ' < "/proc/$TASK_PID/cmdline")"
        echo "gpu_pid=$TASK_PID cmd=$TASK_CMDLINE"
        case "$TASK_CMDLINE" in
            *scripts/gpu_burn.py*) ;;
            *) echo "REFUSE: non-burn GPU process found: pid=$TASK_PID"; exit 1 ;;
        esac
    done
    for TASK_PID in $TASK_GPU_PIDS; do
        kill -TERM "$TASK_PID" 2>/dev/null || true
    done
fi

for TASK_ATTEMPT in 1 2 3 4 5 6; do
    sleep 5
    TASK_REMAINING_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d' | sort -nu)"
    [ -z "$TASK_REMAINING_PIDS" ] && break
    echo "waiting_for_gpu_exit attempt=$TASK_ATTEMPT pids=$TASK_REMAINING_PIDS"
done

TASK_REMAINING_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d' | sort -nu)"
if [ -n "$TASK_REMAINING_PIDS" ]; then
    for TASK_PID in $TASK_REMAINING_PIDS; do
        TASK_CMDLINE="$(tr '\0' ' ' < "/proc/$TASK_PID/cmdline")"
        case "$TASK_CMDLINE" in
            *scripts/gpu_burn.py*) echo "force_kill_burn_pid=$TASK_PID"; kill -KILL "$TASK_PID" ;;
            *) echo "REFUSE: non-burn GPU process remains: pid=$TASK_PID"; exit 1 ;;
        esac
    done
    sleep 5
fi

echo '=== verify all 8 B200 GPUs are free ==='
TASK_FINAL_GPU_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')"
test -z "$TASK_FINAL_GPU_PIDS"
while IFS=',' read -r TASK_INDEX TASK_NAME; do
    TASK_NAME="${TASK_NAME# }"
    echo "gpu=$TASK_INDEX name=$TASK_NAME"
    case "$TASK_NAME" in *B200*) ;; *) echo "ERROR: non-B200 GPU: $TASK_NAME"; exit 1 ;; esac
done < <(nvidia-smi --query-gpu=index,name --format=csv,noheader)
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
echo 'TH2 ALL 8 B200 GPUS FREE'

echo '=== install and verify Accelerate configuration ==='
test -s "$TASK_ACCELERATE_SOURCE"
mkdir -p "$(dirname "$TASK_ACCELERATE_TARGET")"
cp "$TASK_ACCELERATE_SOURCE" "$TASK_ACCELERATE_TARGET"
cmp "$TASK_ACCELERATE_SOURCE" "$TASK_ACCELERATE_TARGET"
grep -Fxq 'distributed_type: MULTI_GPU' "$TASK_ACCELERATE_TARGET"
grep -Fxq 'num_processes: 8' "$TASK_ACCELERATE_TARGET"
echo "accelerate_config=$TASK_ACCELERATE_TARGET"

echo '=== independent low-rank output training preflight ==='
for TASK_REQUIRED_FILE in \
    "$TASK_MODEL_DIR/config.json" \
    "$TASK_MODEL_DIR/tokenizer.json" \
    "$TASK_MODEL_DIR/tokenizer_config.json" \
    "$TASK_PROJECT_DIR/run_experiments.py" \
    "$TASK_PROJECT_DIR/scripts/train_lowrank_independent_output_r128.sh"; do
    test -s "$TASK_REQUIRED_FILE"
done
for TASK_LANG in en vi zh ru de ar; do
    test -d "$TASK_TRAIN_DATA/$TASK_LANG"
done
case "$TASK_OUTPUT_DIR" in
    /mnt/local/_outputs/*/lowrank_independent_output_r128) ;;
    *) echo "REFUSE: unexpected training output path: $TASK_OUTPUT_DIR"; exit 1 ;;
esac
if [ -e "$TASK_OUTPUT_DIR" ]; then
    echo "REFUSE: training output already exists: $TASK_OUTPUT_DIR"
    du -sh "$TASK_OUTPUT_DIR" || true
    exit 1
fi
mkdir -p "$TASK_OUTPUT_BASE"

export SPARSE_EMB_PYTHON="$TASK_PYTHON"
export SPARSE_EMB_MODEL_DIR="$TASK_MODEL_DIR"
export SPARSE_EMB_DATA_DIR="$TASK_TRAIN_DATA"
export SPARSE_EMB_OUTPUT_BASE="$TASK_OUTPUT_BASE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

"$TASK_PYTHON" run_experiments.py --list | tee /tmp/th2-independent-list.txt
grep -Fq '[12] lowrank_independent_output_r128:' /tmp/th2-independent-list.txt
echo 'batch_configuration=16_per_device_x_4_accum_x_8_gpus=512_sequences_per_step'
echo '=== launch lowrank_independent_output_r128 to checkpoint 10000 ==='
"$TASK_PYTHON" -u run_experiments.py \
    --experiments 12 \
    --stop-at-step 10000 \
    --log-dir "$TASK_LOG_DIR"
