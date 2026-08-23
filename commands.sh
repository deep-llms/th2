#1 +60+a
#th2-verify-independent-lr-b200-runtime-20260823-0924
set -euo pipefail

TASK_PYTHON=/mnt/local/conda-py311/envs/sparse_emb/bin/python3.11
TASK_OUTPUT=/mnt/local/_outputs/deep-llms_th2/lowrank_independent_output_r128
TASK_LOG_DIR=/mnt/local/_outputs/deep-llms_th2/logs/lowrank_independent_output_r128
TASK_LOG="$TASK_LOG_DIR/lowrank_independent_output_r128.log"

echo '=== timestamp ==='
date -u

echo '=== B200 state ==='
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader

echo '=== validate exactly eight training GPU workers ==='
mapfile -t TASK_GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' \
        | sort -u
)
if [ "${#TASK_GPU_PIDS[@]}" -ne 8 ]; then
    echo "ERROR: expected 8 unique GPU worker PIDs, found ${#TASK_GPU_PIDS[@]}"
    exit 1
fi
for pid in "${TASK_GPU_PIDS[@]}"; do
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline")
    echo "gpu_worker_pid=$pid cmd=$cmd"
    case "$cmd" in
        *train_compositional.py*) ;;
        *) echo "ERROR: unexpected GPU process $pid"; exit 1 ;;
    esac
done

echo '=== process tree ==='
ps -eo pid,ppid,etimes,%cpu,%mem,args \
    | grep -E 'run_experiments.py|accelerate.commands.launch|train_compositional.py' \
    | grep -v grep

echo '=== validate latest complete checkpoint ==='
test -d "$TASK_OUTPUT"
latest_checkpoint=$(
    find "$TASK_OUTPUT" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' \
        -printf '%f\n' | sort -V | tail -1
)
if [ -z "$latest_checkpoint" ]; then
    echo 'ERROR: no checkpoint exists yet'
    exit 1
fi
checkpoint_dir="$TASK_OUTPUT/$latest_checkpoint"
checkpoint_step=${latest_checkpoint#checkpoint-}
echo "latest_checkpoint=$checkpoint_dir"
for required in config.json model.safetensors trainer_state.json optimizer.pt scheduler.pt \
                embedding.pt output_head.pt rng_state_0.pth rng_state_1.pth \
                rng_state_2.pth rng_state_3.pth rng_state_4.pth rng_state_5.pth \
                rng_state_6.pth rng_state_7.pth; do
    test -s "$checkpoint_dir/$required" || {
        echo "ERROR: missing or empty $checkpoint_dir/$required"
        exit 1
    }
done

"$TASK_PYTHON" - "$checkpoint_dir/trainer_state.json" "$checkpoint_step" <<'PY'
import json
import math
import sys

state_path, expected_step = sys.argv[1], int(sys.argv[2])
with open(state_path) as handle:
    state = json.load(handle)
assert state["global_step"] == expected_step, (state["global_step"], expected_step)
history = state.get("log_history", [])
metric_rows = [row for row in history if "loss" in row]
assert metric_rows, "trainer_state has no loss metrics"
for row in metric_rows:
    for key in ("loss", "grad_norm", "learning_rate"):
        value = row.get(key)
        if value is not None:
            assert isinstance(value, (int, float)) and math.isfinite(value), (key, value)
print("trainer_state_global_step=", state["global_step"], sep="")
print("finite_metric_rows=", len(metric_rows), sep="")
print("latest_metrics=", metric_rows[-1], sep="")
PY

echo '=== validate live training log ==='
test -s "$TASK_LOG"
stat --format='log_size=%s log_modified=%y' "$TASK_LOG"
if grep -E -i 'Traceback|CUDA out of memory|OutOfMemoryError|NCCL[^[:cntrl:]]*(error|failed)|RuntimeError:' "$TASK_LOG"; then
    echo 'ERROR: fatal signature found in training log'
    exit 1
fi
tail -120 "$TASK_LOG"

echo 'TH2 INDEPENDENT LR128 OUTPUT TRAINING HEALTHY'
