#1 +60+a
#th2-check-independent-lr-eta-20260823
set -euo pipefail

TASK_LOG=/mnt/local/_outputs/deep-llms_th2/logs/lowrank_independent_output_r128/lowrank_independent_output_r128.log
TASK_OUTPUT=/mnt/local/_outputs/deep-llms_th2/lowrank_independent_output_r128
TASK_PYTHON=/mnt/local/conda-py311/envs/sparse_emb/bin/python3.11

echo '=== current training state ==='
date -u
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader

mapfile -t TASK_GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -u
)
echo "unique_gpu_workers=${#TASK_GPU_PIDS[@]}"
test "${#TASK_GPU_PIDS[@]}" -eq 8
for pid in "${TASK_GPU_PIDS[@]}"; do
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline")
    case "$cmd" in
        *train_compositional.py*) ;;
        *) echo "ERROR: unexpected GPU process $pid: $cmd"; exit 1 ;;
    esac
done

latest_checkpoint=$(
    find "$TASK_OUTPUT" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' \
        -printf '%f\n' | sort -V | tail -1
)
test -n "$latest_checkpoint"
echo "latest_checkpoint=$latest_checkpoint"

"$TASK_PYTHON" - "$TASK_LOG" "$TASK_OUTPUT/$latest_checkpoint/trainer_state.json" <<'PY'
import datetime
import json
import math
import re
import sys

log_path, state_path = sys.argv[1:]
with open(state_path) as handle:
    state = json.load(handle)
with open(log_path, "rb") as handle:
    handle.seek(0, 2)
    size = handle.tell()
    handle.seek(max(0, size - 4_000_000))
    text = handle.read().decode("utf-8", errors="replace")

progress = re.findall(r"(\d+)/(\d+)[^\r\n]*?([0-9]+(?:\.[0-9]+)?)s/it", text)
assert progress, "no live tqdm progress found"
current_step, scheduled_steps, seconds_per_step = progress[-1]
current_step = int(current_step)
scheduled_steps = int(scheduled_steps)
seconds_per_step = float(seconds_per_step)
target_step = 10_000
assert current_step < target_step, current_step
remaining_seconds = (target_step - current_step) * seconds_per_step + 120
finish = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=remaining_seconds)

metrics = [row for row in state.get("log_history", []) if "loss" in row]
assert metrics
latest = metrics[-1]
for key in ("loss", "grad_norm", "learning_rate"):
    assert math.isfinite(float(latest[key])), (key, latest[key])

print(f"checkpoint_global_step={state['global_step']}")
print(f"live_step={current_step}")
print(f"full_schedule_steps={scheduled_steps}")
print(f"seconds_per_step={seconds_per_step:.3f}")
print(f"target_step={target_step}")
print(f"estimated_remaining_hours={remaining_seconds / 3600:.2f}")
print(f"estimated_finish_utc={finish:%Y-%m-%d %H:%M:%S UTC}")
print(f"latest_checkpoint_metrics={latest}")
PY

echo 'TH2 TRAINING ETA CHECK OK'
