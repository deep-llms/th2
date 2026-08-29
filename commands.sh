#1 +30+a
#th2-verify-dense-ddp-default-completion-and-burn-20260829-a01
set -euo pipefail

TASK_PYTHON=/mnt/local/conda-py311/envs/sparse_emb/bin/python3.11
TASK_OUTPUT=/mnt/local/_outputs/@PROJECT@/dense_tied_baseline_b200_ddp_default
TASK_CHECKPOINT="$TASK_OUTPUT/checkpoint-10000"
TASK_LOG_DIR=/mnt/local/_outputs/@PROJECT@/logs/dense_tied_baseline_b200_ddp_default_20260829
TASK_EXPERIMENT_LOG="$TASK_LOG_DIR/experiments.log"
TASK_TRAIN_LOG="$TASK_LOG_DIR/dense_tied_baseline_b200_ddp_default.log"

echo '=== identity ==='
date -u
hostname
git rev-parse HEAD

echo '=== verify completed dense DDP-default checkpoint ==='
test -x "$TASK_PYTHON"
test -s "$TASK_EXPERIMENT_LOG"
test -s "$TASK_TRAIN_LOG"
grep -F 'dense_tied_baseline_b200_ddp_default: STOPPED at step 10000' \
    "$TASK_EXPERIMENT_LOG"
for TASK_FILE in \
    config.json model.safetensors trainer_state.json optimizer.pt scheduler.pt \
    rng_state_0.pth rng_state_1.pth rng_state_2.pth rng_state_3.pth \
    rng_state_4.pth rng_state_5.pth rng_state_6.pth rng_state_7.pth; do
    test -s "$TASK_CHECKPOINT/$TASK_FILE"
done
"$TASK_PYTHON" - "$TASK_CHECKPOINT" <<'PY'
import json
import os
import sys

checkpoint = sys.argv[1]
with open(os.path.join(checkpoint, "trainer_state.json"), encoding="utf-8") as handle:
    trainer_state = json.load(handle)
with open(os.path.join(checkpoint, "config.json"), encoding="utf-8") as handle:
    config = json.load(handle)
assert int(trainer_state["global_step"]) == 10000, trainer_state["global_step"]
assert config.get("model_type") == "qwen3", config.get("model_type")
assert config.get("tie_word_embeddings") is True, config.get("tie_word_embeddings")
print("DENSE_DDP_DEFAULT_CHECKPOINT_OK step=10000 native_tied=true")
PY
if grep -HniE 'Traceback|CUDA out of memory|OutOfMemoryError|NCCL.*(unhandled|system error|remote process exited|watchdog|timeout)|Segmentation fault|Bus error' \
        "$TASK_TRAIN_LOG" "$TASK_EXPERIMENT_LOG"; then
    echo 'ERROR: fatal signature found in dense retrain logs' >&2
    exit 1
fi
tail -c 160000 "$TASK_TRAIN_LOG" | tr '\r' '\n' \
    | grep -E "\{'loss':|train_runtime|train_loss" | tail -20 || true
du -sh "$TASK_OUTPUT"

echo '=== verify no training remains ==='
if pgrep -af '[t]rain.py|[r]un_experiments.py|[a]ccelerate.commands.launch|[a]ccelerate launch'; then
    echo 'ERROR: a training or Accelerate process remains' >&2
    exit 1
fi

echo '=== inspect current GPU ownership ==='
mapfile -t TASK_GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -nu
)
test "${#TASK_GPU_PIDS[@]}" -eq 8
for TASK_PID in "${TASK_GPU_PIDS[@]}"; do
    TASK_CMDLINE="$(tr '\0' ' ' < "/proc/$TASK_PID/cmdline")"
    echo "gpu_pid=$TASK_PID cmdline=$TASK_CMDLINE"
    case "$TASK_CMDLINE" in
        *scripts/gpu_burn.py*) ;;
        *) echo "ERROR: unexpected GPU process $TASK_PID" >&2; exit 1 ;;
    esac
done
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,power.draw \
    --format=csv,noheader
echo 'TH2 DENSE RETRAIN AND PROJECT GPU BURNS VERIFIED'
