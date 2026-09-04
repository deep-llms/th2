#1 +180+a
#th2-readonly-audit-tiered-c512-live-training-20260904-a02
set -euo pipefail

die() {
    echo "ERROR: $*" >&2
    exit 1
}

TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_TIERED="$TASK_OUTPUT_BASE/tiered_ranklift_lb_t4_c512"
TASK_CONTROL="$TASK_OUTPUT_BASE/groupreduce_matched_lb_t4"
TASK_LOG_DIR="$TASK_OUTPUT_BASE/logs/tiered_c512_lb_groupreduce_10k_20260904_a01"
TASK_TIERED_LOG="$TASK_LOG_DIR/tiered_ranklift_lb_t4_c512.log"

echo '=== read-only identity and GPU snapshot ==='
date -u
hostname
nvidia-smi \
    --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader
nvidia-smi \
    --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits

echo '=== prove exactly one active training rank per GPU ==='
TASK_ALL_GPU_PIDS=()
for TASK_GPU_INDEX in 0 1 2 3 4 5 6 7; do
    mapfile -t TASK_GPU_PIDS < <(
        nvidia-smi -i "$TASK_GPU_INDEX" \
            --query-compute-apps=pid --format=csv,noheader,nounits |
            sed 's/^[[:space:]]*//;s/[[:space:]]*$//;/^$/d' | sort -nu
    )
    [[ "${#TASK_GPU_PIDS[@]}" -eq 1 ]] \
        || die "GPU $TASK_GPU_INDEX has ${#TASK_GPU_PIDS[@]} compute processes, expected one"
    TASK_PID="${TASK_GPU_PIDS[0]}"
    [[ "$TASK_PID" =~ ^[0-9]+$ && "$TASK_PID" -ne 1 ]] \
        || die "invalid GPU PID on GPU $TASK_GPU_INDEX: $TASK_PID"
    TASK_CMDLINE="$(tr '\0' ' ' < "/proc/$TASK_PID/cmdline")"
    [[ "$TASK_CMDLINE" == *'train_compositional.py'* ]] \
        || die "GPU $TASK_GPU_INDEX is not held by train_compositional.py: $TASK_CMDLINE"
    [[ "$TASK_CMDLINE" == *'--arm tiered_ranklift'* ]] \
        || die "GPU $TASK_GPU_INDEX is not running Tiered RankLift"
    [[ "$TASK_CMDLINE" == *'--tiered_ranklift_code_dims 1024,512,192,64'* ]] \
        || die "wrong code dimensions on GPU $TASK_GPU_INDEX"
    [[ "$TASK_CMDLINE" == *'--tiered_ranklift_lift_dims 0,0,320,192'* ]] \
        || die "wrong lift dimensions on GPU $TASK_GPU_INDEX"
    [[ "$TASK_CMDLINE" == *'--tiered_ranklift_populations 2048,6144,24576,119168'* ]] \
        || die "wrong populations on GPU $TASK_GPU_INDEX"
    [[ "$TASK_CMDLINE" == *'--tie_output'* ]] \
        || die "tied output is absent on GPU $TASK_GPU_INDEX"
    [[ "$TASK_CMDLINE" == *'--bf16'* ]] \
        || die "BF16 is absent on GPU $TASK_GPU_INDEX"
    [[ "$TASK_CMDLINE" == *'--per_device_train_batch_size 16'* ]] \
        || die "wrong per-device batch on GPU $TASK_GPU_INDEX"
    [[ "$TASK_CMDLINE" == *'--gradient_accumulation_steps 4'* ]] \
        || die "wrong gradient accumulation on GPU $TASK_GPU_INDEX"
    [[ "$TASK_CMDLINE" == *'--ddp_find_unused_parameters false'* ]] \
        || die "wrong DDP unused-parameter setting on GPU $TASK_GPU_INDEX"
    [[ "$TASK_CMDLINE" != *'--max_steps'* ]] \
        || die "production command unexpectedly sets max_steps"
    TASK_ALL_GPU_PIDS+=("$TASK_PID")
    echo "gpu=$TASK_GPU_INDEX training_pid=$TASK_PID"
done
mapfile -t TASK_UNIQUE_GPU_PIDS < <(
    printf '%s\n' "${TASK_ALL_GPU_PIDS[@]}" | sort -nu
)
[[ "${#TASK_UNIQUE_GPU_PIDS[@]}" -eq 8 ]] \
    || die 'the eight GPUs do not have eight distinct training ranks'

echo '=== process hierarchy (read only) ==='
pgrep -af '[r]un_experiments.py|[a]ccelerate.commands.launch|[t]rain_compositional.py' || true
[[ "$(pgrep -fc '[r]un_experiments.py')" -eq 1 ]] \
    || die 'expected exactly one experiment runner'
[[ "$(pgrep -fc '[a]ccelerate.commands.launch')" -eq 1 ]] \
    || die 'expected exactly one Accelerate launcher'

echo '=== validate persisted experiment configuration ==='
test -s "$TASK_TIERED/train_config.json"
test -s "$TASK_TIERED_LOG"
test -s "$TASK_LOG_DIR/experiments.log"
/mnt/local/conda-py311/envs/sparse_emb/bin/python3.11 - \
    "$TASK_TIERED/train_config.json" "$TASK_TIERED" "$TASK_TIERED_LOG" <<'PY'
import csv
import json
import math
import pathlib
import re
import sys

config_path = pathlib.Path(sys.argv[1])
output_dir = pathlib.Path(sys.argv[2])
log_path = pathlib.Path(sys.argv[3])
config = json.loads(config_path.read_text())
comp = config["compositional"]
training = config["training"]

assert comp["arm"] == "tiered_ranklift", comp
assert comp["tie_output"] is True, comp
assert comp["tiered_ranklift_code_dims"] == "1024,512,192,64", comp
assert comp["tiered_ranklift_lift_dims"] == "0,0,320,192", comp
assert comp["tiered_ranklift_populations"] == "2048,6144,24576,119168", comp
assert training["bf16"] is True, training
assert training["per_device_train_batch_size"] == 16, training
assert training["gradient_accumulation_steps"] == 4, training
assert training["ddp_find_unused_parameters"] is False, training
assert training["seed"] == 42, training
assert training["save_steps"] == 250, training

text = log_path.read_text(errors="replace")
fatal = (
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "OutOfMemoryError",
    "ChildFailedError",
    "ProcessExitedException",
    "Segmentation fault",
    "Bus error",
)
found = [marker for marker in fatal if marker in text]
assert not found, found

losses = [
    float(value)
    for value in re.findall(r"'loss': ['\"]?([0-9.eE+-]+)", text)
]
assert losses and all(math.isfinite(value) for value in losses), losses[-10:]

checkpoints = []
for path in output_dir.glob("checkpoint-*"):
    try:
        step = int(path.name.removeprefix("checkpoint-"))
    except ValueError:
        continue
    state_path = path / "trainer_state.json"
    if state_path.is_file() and state_path.stat().st_size:
        state = json.loads(state_path.read_text())
        assert int(state["global_step"]) == step, (path, state["global_step"])
        checkpoints.append(step)

progress_path = output_dir / "train_progress.csv"
progress_rows = []
if progress_path.is_file() and progress_path.stat().st_size:
    with progress_path.open(newline="") as handle:
        progress_rows = list(csv.DictReader(handle))

print("PERSISTED_CONFIG_OK")
print(
    "FINITE_LOSSES_OK",
    f"count={len(losses)}",
    f"first={losses[0]:.6f}",
    f"last={losses[-1]:.6f}",
    f"minimum={min(losses):.6f}",
)
print("VALID_CHECKPOINT_STEPS", checkpoints[-8:])
if progress_rows:
    print("LATEST_PROGRESS_ROW", progress_rows[-1])
else:
    print("LATEST_PROGRESS_ROW unavailable")
PY

echo '=== confirm control remains queued, not accidentally concurrent ==='
if [[ -e "$TASK_CONTROL" ]]; then
    find "$TASK_CONTROL" -mindepth 1 -maxdepth 2 -printf '%P\n' | head -50
    die 'control output exists while Tiered is still the active GPU workload'
fi
echo 'CONTROL_CORRECTLY_QUEUED_NOT_STARTED'

echo '=== current progress and log tails ==='
find "$TASK_TIERED" -mindepth 1 -maxdepth 1 -type d \
    -name 'checkpoint-*' -printf '%f\n' | sort -V | tail -8 || true
if [[ -s "$TASK_TIERED/train_progress.csv" ]]; then
    tail -8 "$TASK_TIERED/train_progress.csv"
fi
tail -30 "$TASK_LOG_DIR/experiments.log"
tail -80 "$TASK_TIERED_LOG"

echo 'TH2 LIVE TIERED-C512 TRAINING READ-ONLY AUDIT PASSED'
