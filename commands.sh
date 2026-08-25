#1 +120+a
#th2-stop-burn-launch-pure-local-g16-r128-10k-20260825-retry1
set -euo pipefail

TASK_PROJECT_DIR="$PWD"
TASK_CONDA=/mnt/local/conda-py311/bin/conda
TASK_PYTHON=/mnt/local/conda-py311/envs/sparse_emb/bin/python3.11
TASK_MODEL_DIR=/mnt/local/_models/@PROJECT@/Qwen3-0.6B
TASK_DATA_DIR=/mnt/local/_data/@PROJECT@/data/Qwen_Qwen3-0.6B/train
TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_OUTPUT_DIR="$TASK_OUTPUT_BASE/pure_local_tied_g16_r128"
TASK_LOG_DIR="$TASK_OUTPUT_BASE/logs/pure_local_tied_g16_r128_20260825"
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

echo '=== identify and stop only supervised project GPU burns ==='
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
mapfile -t TASK_GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -u
)
if [ "${#TASK_GPU_PIDS[@]}" -gt 0 ]; then
    for TASK_PID in "${TASK_GPU_PIDS[@]}"; do
        TASK_CMDLINE="$(tr '\0' ' ' < "/proc/$TASK_PID/cmdline")"
        echo "gpu_pid=$TASK_PID cmd=$TASK_CMDLINE"
        case "$TASK_CMDLINE" in
            *scripts/gpu_burn.py*) ;;
            *)
                echo "REFUSE: GPU PID $TASK_PID is not the project burn"
                exit 1
                ;;
        esac
    done
    kill -TERM "${TASK_GPU_PIDS[@]}" 2>/dev/null || true
    for TASK_WAIT in $(seq 1 30); do
        TASK_ALIVE=0
        for TASK_PID in "${TASK_GPU_PIDS[@]}"; do
            kill -0 "$TASK_PID" 2>/dev/null && TASK_ALIVE=$((TASK_ALIVE + 1))
        done
        [ "$TASK_ALIVE" -eq 0 ] && break
        sleep 1
    done
    for TASK_PID in "${TASK_GPU_PIDS[@]}"; do
        if kill -0 "$TASK_PID" 2>/dev/null; then
            TASK_CMDLINE="$(tr '\0' ' ' < "/proc/$TASK_PID/cmdline")"
            case "$TASK_CMDLINE" in
                *scripts/gpu_burn.py*) kill -KILL "$TASK_PID" ;;
                *) echo "REFUSE: PID $TASK_PID changed identity"; exit 1 ;;
            esac
        fi
    done
fi
sleep 5

echo '=== verify exactly eight free B200 GPUs ==='
mapfile -t TASK_GPU_NAMES < <(
    nvidia-smi --query-gpu=name --format=csv,noheader \
        | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
)
test "${#TASK_GPU_NAMES[@]}" -eq 8
for TASK_GPU in "${!TASK_GPU_NAMES[@]}"; do
    echo "gpu=$TASK_GPU name=${TASK_GPU_NAMES[$TASK_GPU]}"
    case "${TASK_GPU_NAMES[$TASK_GPU]}" in
        *B200*) ;;
        *) echo "ERROR: GPU $TASK_GPU is not a B200"; exit 1 ;;
    esac
done
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
mapfile -t TASK_REMAINING_GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -u
)
test "${#TASK_REMAINING_GPU_PIDS[@]}" -eq 0 || {
    echo "ERROR: GPU processes remain: ${TASK_REMAINING_GPU_PIDS[*]}"
    exit 1
}
echo 'TH2 ALL 8 B200 GPUS FREE AFTER STOPPING BURNS'

echo '=== verify model, data, and fresh experiment paths ==='
test -s "$TASK_MODEL_DIR/config.json"
test -s "$TASK_MODEL_DIR/tokenizer.json"
for TASK_LANG in en vi zh ru de ar; do
    test -d "$TASK_DATA_DIR/$TASK_LANG"
    TASK_ARROW_COUNT=$(find "$TASK_DATA_DIR/$TASK_LANG" -type f -name '*.arrow' | wc -l)
    test "$TASK_ARROW_COUNT" -gt 0
    echo "source_language=$TASK_LANG arrow_files=$TASK_ARROW_COUNT"
done
case "$TASK_OUTPUT_DIR" in
    /mnt/local/_outputs/*/pure_local_tied_g16_r128) ;;
    *) echo "REFUSE: unexpected output path: $TASK_OUTPUT_DIR"; exit 1 ;;
esac
for TASK_PATH in "$TASK_OUTPUT_DIR" "$TASK_LOG_DIR"; do
    test ! -e "$TASK_PATH" || {
        echo "REFUSE: stale Pure-local path exists: $TASK_PATH"
        du -sh "$TASK_PATH" || true
        exit 1
    }
    echo "fresh_path=$TASK_PATH"
done
TASK_CACHE_COUNT=$(find "$TASK_DATA_DIR" -type f -name 'cache-*' | wc -l)
echo "existing_dataset_cache_files=$TASK_CACHE_COUNT (reused; source data unchanged)"

"$TASK_PYTHON" - "$TASK_MODEL_DIR/config.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    config = json.load(handle)
assert config.get("model_type") == "qwen3", config.get("model_type")
assert config.get("hidden_size") == 1024, config.get("hidden_size")
assert config.get("vocab_size") == 151936, config.get("vocab_size")
print("model_config=Qwen3-0.6B hidden=1024 vocab=151936")
PY

echo '=== install and verify eight-GPU bf16 Accelerate configuration ==='
test -s "$TASK_ACCELERATE_SOURCE"
mkdir -p "$(dirname "$TASK_ACCELERATE_DEST")"
cp "$TASK_ACCELERATE_SOURCE" "$TASK_ACCELERATE_DEST"
cmp "$TASK_ACCELERATE_SOURCE" "$TASK_ACCELERATE_DEST"
grep -Fx 'distributed_type: MULTI_GPU' "$TASK_ACCELERATE_DEST"
grep -Fx 'mixed_precision: bf16' "$TASK_ACCELERATE_DEST"
grep -Fx 'num_processes: 8' "$TASK_ACCELERATE_DEST"
echo "accelerate_config=$TASK_ACCELERATE_DEST"

echo '=== verify Pure-local code and experiment definition ==='
export SPARSE_EMB_PYTHON="$TASK_PYTHON"
export SPARSE_EMB_MODEL_DIR="$TASK_MODEL_DIR"
export SPARSE_EMB_DATA_DIR="$TASK_DATA_DIR"
export SPARSE_EMB_OUTPUT_BASE="$TASK_OUTPUT_BASE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
"$TASK_PYTHON" -u run_experiments.py --list \
    | grep -F '[17] pure_local_tied_g16_r128:'
grep -F -- '--arm pure_local' scripts/train_pure_local_tied_g16_r128.sh
grep -F -- '--pure_local_rank 128' scripts/train_pure_local_tied_g16_r128.sh
grep -F -- '--num_groups 16' scripts/train_pure_local_tied_g16_r128.sh
grep -F -- '--tie_output' scripts/train_pure_local_tied_g16_r128.sh
"$TASK_PYTHON" - <<'PY'
from train_compositional import CompositionalArguments, build_arm

args = CompositionalArguments(
    arm="pure_local", pure_local_rank=128, num_groups=16, tie_output=True
)
embed = build_arm(args, vocab_size=151936, embed_dim=1024)
assert embed.group_size == 9496
assert embed.num_large_groups == 0
assert sum(parameter.numel() for parameter in embed.parameters()) == 21545984
print("pure_local_preflight=OK params=21545984 group_size=9496")
PY
echo 'architecture=pure_local rank=128 groups=16 exact_tied_output=true'
echo 'batch_configuration=16_per_device_x_4_accum_x_8_gpus=512_sequences_per_step'
echo 'precision=bf16 target_checkpoint=10000 original_one_epoch_schedule=true'

echo '=== launch Pure-local G16 R128 to checkpoint 10000 ==='
exec "$TASK_PYTHON" -u run_experiments.py \
    --experiments 17 \
    --stop-at-step 10000 \
    --log-dir "$TASK_LOG_DIR"
