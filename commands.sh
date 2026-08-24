#1 +60+a
#th2-check-global-lowrank-tied-r128-b200-runtime-20260824
set -euo pipefail

TASK_OUTPUT=/mnt/local/_outputs/@PROJECT@/global_lowrank_tied_r128_b200
TASK_LOG=/mnt/local/_outputs/@PROJECT@/logs/b200_matched_global_lr_then_dense_tied_20260824/global_lowrank_tied_r128_b200.log
TASK_PYTHON=/mnt/local/conda-py311/envs/sparse_emb/bin/python3.11

echo '=== timestamp and B200 utilization ==='
date -u
hostname
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader

echo '=== verify exactly eight global-LR tied workers ==='
mapfile -t TASK_GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -u
)
test "${#TASK_GPU_PIDS[@]}" -eq 8 || {
    echo "ERROR: expected 8 unique GPU workers, found ${#TASK_GPU_PIDS[@]}"
    exit 1
}
for pid in "${TASK_GPU_PIDS[@]}"; do
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline")
    [[ "$cmd" == *train_compositional.py* ]] || { echo "ERROR: wrong worker $pid: $cmd"; exit 1; }
    [[ "$cmd" == *"--arm lowrank"* ]] || { echo "ERROR: missing lowrank arm for $pid"; exit 1; }
    [[ "$cmd" == *"--d_x 128"* ]] || { echo "ERROR: missing rank 128 for $pid"; exit 1; }
    [[ "$cmd" == *"--tie_output"* ]] || { echo "ERROR: missing exact tying for $pid"; exit 1; }
    [[ "$cmd" == *"--output_dir $TASK_OUTPUT"* ]] || { echo "ERROR: wrong output for $pid"; exit 1; }
    echo "global_lr_tied_gpu_worker=$pid"
done

echo '=== verify live training log ==='
test -s "$TASK_LOG"
stat --format='log_size=%s log_modified=%y' "$TASK_LOG"
grep -F 'Output tied to input embedding (lm_head replaced)' "$TASK_LOG"
grep -F 'Training new model from scratch' "$TASK_LOG" | tail -1
if grep -E -i 'Traceback|CUDA out of memory|OutOfMemoryError|RuntimeError:|NCCL[^[:cntrl:]]*(unhandled|system error|remote process exited|watchdog|collective operation timeout)' "$TASK_LOG"; then
    echo 'ERROR: fatal signature found in training log'
    exit 1
fi
tail -40 "$TASK_LOG"

echo '=== latest checkpoint if available ==='
latest_checkpoint=$(
    find "$TASK_OUTPUT" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' \
        -printf '%f\n' 2>/dev/null | sort -V | tail -1
)
if [ -n "$latest_checkpoint" ]; then
    checkpoint_dir="$TASK_OUTPUT/$latest_checkpoint"
    checkpoint_step=${latest_checkpoint#checkpoint-}
    for required in config.json model.safetensors trainer_state.json optimizer.pt scheduler.pt \
                    embedding.pt rng_state_0.pth rng_state_1.pth rng_state_2.pth \
                    rng_state_3.pth rng_state_4.pth rng_state_5.pth rng_state_6.pth \
                    rng_state_7.pth; do
        test -s "$checkpoint_dir/$required"
    done
    "$TASK_PYTHON" - "$checkpoint_dir/trainer_state.json" "$checkpoint_step" <<'PY'
import json
import math
import sys

path, expected = sys.argv[1], int(sys.argv[2])
with open(path) as handle:
    state = json.load(handle)
assert state["global_step"] == expected, (state["global_step"], expected)
metrics = [row for row in state.get("log_history", []) if "loss" in row]
assert metrics
for row in metrics:
    for key in ("loss", "grad_norm", "learning_rate"):
        assert math.isfinite(float(row[key])), (key, row[key])
print(f"latest_checkpoint=checkpoint-{expected}")
print(f"latest_metrics={metrics[-1]}")
PY
else
    echo 'latest_checkpoint=not_created_yet'
fi

echo 'TH2 GLOBAL LOWRANK TIED R128 B200 TRAINING HEALTHY'
