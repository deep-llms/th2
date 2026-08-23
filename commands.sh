#1 +60+a
#th2-verify-shared-local-tied-g16-runtime-20260823
set -euo pipefail

TASK_OUTPUT=/mnt/local/_outputs/@PROJECT@/shared_local_tied_g16
TASK_LOG=/mnt/local/_outputs/@PROJECT@/logs/shared_local_tied_g16/shared_local_tied_g16.log
TASK_PYTHON=/mnt/local/conda-py311/envs/sparse_emb/bin/python3.11

echo '=== timestamp and B200 state ==='
date -u
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader

echo '=== validate exactly eight G16 training GPU workers ==='
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
    case "$cmd" in
        *train_compositional.py*"--arm shared_local"*"--shared_rank 64"*"--local_embed_rank 64"*"--num_groups 16"*"--tie_output"*) ;;
        *) echo "ERROR: unexpected GPU worker $pid: $cmd"; exit 1 ;;
    esac
    echo "g16_gpu_worker=$pid"
done

echo '=== validate latest complete checkpoint ==='
latest_checkpoint=$(
    find "$TASK_OUTPUT" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' \
        -printf '%f\n' | sort -V | tail -1
)
test -n "$latest_checkpoint" || {
    echo 'ERROR: no checkpoint exists yet'
    exit 1
}
checkpoint_dir="$TASK_OUTPUT/$latest_checkpoint"
checkpoint_step=${latest_checkpoint#checkpoint-}
echo "latest_checkpoint=$latest_checkpoint"
for required in config.json model.safetensors trainer_state.json optimizer.pt scheduler.pt \
                embedding.pt rng_state_0.pth rng_state_1.pth rng_state_2.pth \
                rng_state_3.pth rng_state_4.pth rng_state_5.pth rng_state_6.pth \
                rng_state_7.pth; do
    test -s "$checkpoint_dir/$required" || {
        echo "ERROR: missing or empty $checkpoint_dir/$required"
        exit 1
    }
done

"$TASK_PYTHON" - "$checkpoint_dir/trainer_state.json" "$checkpoint_step" "$TASK_LOG" <<'PY'
import json
import math
import re
import sys

state_path, expected_step, log_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
with open(state_path) as handle:
    state = json.load(handle)
assert state["global_step"] == expected_step, (state["global_step"], expected_step)
metrics = [row for row in state.get("log_history", []) if "loss" in row]
assert metrics, "checkpoint has no loss metrics"
for row in metrics:
    for key in ("loss", "grad_norm", "learning_rate"):
        assert math.isfinite(float(row[key])), (key, row[key])
with open(log_path, "rb") as handle:
    handle.seek(0, 2)
    size = handle.tell()
    handle.seek(max(0, size - 4_000_000))
    text = handle.read().decode("utf-8", errors="replace")
progress = re.findall(r"(\d+)/(\d+)[^\r\n]*?([0-9]+(?:\.[0-9]+)?)s/it", text)
assert progress, "no live tqdm progress found"
live_step, scheduled_steps, seconds_per_step = progress[-1]
print(f"checkpoint_global_step={state['global_step']}")
print(f"finite_metric_rows={len(metrics)}")
print(f"latest_metrics={metrics[-1]}")
print(f"live_step={live_step}/{scheduled_steps}")
print(f"seconds_per_step={float(seconds_per_step):.3f}")
PY

echo '=== validate live log has no fatal training signature ==='
test -s "$TASK_LOG"
stat --format='log_size=%s log_modified=%y' "$TASK_LOG"
if grep -E -i 'Traceback|CUDA out of memory|OutOfMemoryError|RuntimeError:|NCCL[^[:cntrl:]]*(unhandled|system error|remote process exited|watchdog|collective operation timeout)' "$TASK_LOG"; then
    echo 'ERROR: fatal signature found in training log'
    exit 1
fi
echo 'TH2 SHARED LOCAL TIED G16 TRAINING HEALTHY'
