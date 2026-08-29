#1 +60+a
#th2-watch-nested-phase1-then-supervise-burns-20260829-a01
set -euo pipefail

die() {
    echo "ERROR: $*" >&2
    exit 1
}

gpu_pids() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -nu
}

validate_b200_node() {
    local index
    mapfile -t TASK_GPU_NAMES < <(
        nvidia-smi --query-gpu=name --format=csv,noheader \
            | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
    )
    [[ "${#TASK_GPU_NAMES[@]}" -eq 8 ]] \
        || die "expected 8 GPUs, found ${#TASK_GPU_NAMES[@]}"
    for index in "${!TASK_GPU_NAMES[@]}"; do
        [[ "${TASK_GPU_NAMES[$index]}" == *B200* ]] \
            || die "GPU $index is not B200: ${TASK_GPU_NAMES[$index]}"
    done
}

TASK_PROJECT_DIR=/mnt/local/@PROJECT@
TASK_PYTHON=/mnt/local/conda-py311/envs/sparse_emb/bin/python3.11
TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_LOG_DIR="$TASK_OUTPUT_BASE/logs/nested_ladder_phase1_20260829"
TASK_EXPERIMENT_LOG="$TASK_LOG_DIR/experiments.log"
TASK_BURN_SCRIPT="$TASK_PROJECT_DIR/scripts/gpu_burn.py"
TASK_NAMES=(nested_ladder_tied_t4 groupreduce_matched_nested_tied_t4)
TASK_LOCK_DIR="$TASK_OUTPUT_BASE/.locks"
TASK_LOCK_FILE="$TASK_LOCK_DIR/nested_phase1_then_burn.lock"
TASK_EXPECTED_RUNNER_FRAGMENT="run_experiments.py --experiments 21 22 --stop-at-step 10000 --log-dir $TASK_LOG_DIR"

echo '=== nested Phase-1 completion -> burn watcher ==='
date -u
hostname
cd "$TASK_PROJECT_DIR"
test -x "$TASK_PYTHON"
test -s "$TASK_BURN_SCRIPT"
test -s "$TASK_EXPERIMENT_LOG"
validate_b200_node

mkdir -p "$TASK_LOCK_DIR"
exec 9>"$TASK_LOCK_FILE"
flock -n 9 || die "another nested Phase-1 burn watcher already holds $TASK_LOCK_FILE"

echo '=== identify the exact active sequential runner ==='
TASK_RUNNER_PIDS=()
while read -r TASK_PID; do
    [ -n "$TASK_PID" ] || continue
    TASK_CMDLINE="$(tr '\0' ' ' < "/proc/$TASK_PID/cmdline")"
    if [[ "$TASK_CMDLINE" == *"$TASK_EXPECTED_RUNNER_FRAGMENT"* ]]; then
        TASK_RUNNER_PIDS+=("$TASK_PID")
        echo "matched_runner_pid=$TASK_PID cmdline=$TASK_CMDLINE"
    fi
done < <(pgrep -f '[r]un_experiments.py' || true)
[[ "${#TASK_RUNNER_PIDS[@]}" -eq 1 ]] \
    || die "expected exactly one matching sequential runner, found ${#TASK_RUNNER_PIDS[@]}"
TASK_RUNNER_PID="${TASK_RUNNER_PIDS[0]}"

echo '=== wait without modifying the active training run ==='
TASK_POLL_COUNT=0
while kill -0 "$TASK_RUNNER_PID" 2>/dev/null; do
    TASK_CMDLINE="$(tr '\0' ' ' < "/proc/$TASK_RUNNER_PID/cmdline")"
    [[ "$TASK_CMDLINE" == *"$TASK_EXPECTED_RUNNER_FRAGMENT"* ]] \
        || die "runner PID changed identity: $TASK_RUNNER_PID $TASK_CMDLINE"
    if (( TASK_POLL_COUNT % 20 == 0 )); then
        date -u
        echo "runner_pid=$TASK_RUNNER_PID still_active=true"
        tail -12 "$TASK_EXPERIMENT_LOG" || true
    fi
    TASK_POLL_COUNT=$((TASK_POLL_COUNT + 1))
    sleep 30
done

echo '=== runner exited; wait 60 seconds before completion checks ==='
date -u
sleep 60

echo '=== verify both checkpoint-10000 stops and artifacts ==='
for TASK_NAME in "${TASK_NAMES[@]}"; do
    TASK_OUTPUT="$TASK_OUTPUT_BASE/$TASK_NAME"
    TASK_CHECKPOINT="$TASK_OUTPUT/checkpoint-10000"
    TASK_TRAIN_LOG="$TASK_LOG_DIR/$TASK_NAME.log"
    grep -Fq "$TASK_NAME: STOPPED at step 10000" "$TASK_EXPERIMENT_LOG" \
        || die "missing successful stop marker for $TASK_NAME"
    for TASK_FILE in \
        config.json model.safetensors trainer_state.json optimizer.pt \
        scheduler.pt embedding.pt rng_state_0.pth rng_state_1.pth \
        rng_state_2.pth rng_state_3.pth rng_state_4.pth rng_state_5.pth \
        rng_state_6.pth rng_state_7.pth; do
        test -s "$TASK_CHECKPOINT/$TASK_FILE" \
            || die "missing checkpoint artifact: $TASK_CHECKPOINT/$TASK_FILE"
    done
    test -s "$TASK_TRAIN_LOG" || die "missing training log for $TASK_NAME"
    if grep -HniE 'CUDA out of memory|OutOfMemoryError|NCCL.*(unhandled|system error|remote process exited|watchdog|timeout)|Segmentation fault|Bus error' \
            "$TASK_TRAIN_LOG"; then
        die "fatal signature found for $TASK_NAME"
    fi
done

"$TASK_PYTHON" - "$TASK_OUTPUT_BASE" <<'PY'
import json
import os
import sys

base = sys.argv[1]
expected = {
    "nested_ladder_tied_t4": {
        "arm": "nested_ladder",
        "nested_tier_ranks": "64,128,320,512",
        "nested_tier_populations": "151936,32768,8192,2048",
    },
    "groupreduce_matched_nested_tied_t4": {
        "arm": "groupreduce",
        "groupreduce_ranks": "1024,512,192,64",
        "groupreduce_populations": "2048,6144,24576,119168",
    },
}
for name, fields in expected.items():
    checkpoint = os.path.join(base, name, "checkpoint-10000")
    with open(os.path.join(checkpoint, "trainer_state.json"), encoding="utf-8") as handle:
        state = json.load(handle)
    with open(os.path.join(checkpoint, "config.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    comp = config.get("compositional", {})
    assert int(state["global_step"]) == 10000, (name, state["global_step"])
    assert comp.get("tie_output") is True, (name, comp.get("tie_output"))
    for key, value in fields.items():
        assert comp.get(key) == value, (name, key, comp.get(key), value)
    print(f"CHECKPOINT_OK name={name} step=10000 tied=true")
PY

echo '=== prove training is gone and GPUs are free ==='
if pgrep -af '[t]rain_compositional.py|[r]un_experiments.py|[a]ccelerate.commands.launch|[a]ccelerate launch'; then
    die 'training or Accelerate process remains after successful completion'
fi
mapfile -t TASK_REMAINING_GPU_PIDS < <(gpu_pids)
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,power.draw \
    --format=csv,noheader
[[ "${#TASK_REMAINING_GPU_PIDS[@]}" -eq 0 ]] \
    || die "GPU processes remain before burn: ${TASK_REMAINING_GPU_PIDS[*]}"
echo 'ALL 8 B200 GPUS FREE AFTER BOTH PHASE-1 EXPERIMENTS'

echo '=== launch one supervised project burn worker per GPU ==='
TASK_CHILD_PIDS=()
cleanup_burn_workers() {
    trap - EXIT INT TERM
    if [[ "${#TASK_CHILD_PIDS[@]}" -gt 0 ]]; then
        kill -TERM "${TASK_CHILD_PIDS[@]}" 2>/dev/null || true
        wait "${TASK_CHILD_PIDS[@]}" 2>/dev/null || true
    fi
}
trap cleanup_burn_workers EXIT INT TERM

for TASK_GPU in 0 1 2 3 4 5 6 7; do
    TASK_BURN_LOG="/tmp/project_gpu_burn_nested_phase1_gpu${TASK_GPU}.log"
    env CUDA_VISIBLE_DEVICES="$TASK_GPU" "$TASK_PYTHON" -u "$TASK_BURN_SCRIPT" \
        >"$TASK_BURN_LOG" 2>&1 &
    TASK_CHILD_PIDS+=("$!")
    echo "launched_burn gpu=$TASK_GPU pid=$! log=$TASK_BURN_LOG"
done

sleep 30
declare -A TASK_EXPECTED_BURN_PIDS=()
for TASK_PID in "${TASK_CHILD_PIDS[@]}"; do
    kill -0 "$TASK_PID" || die "burn worker exited early: $TASK_PID"
    TASK_EXPECTED_BURN_PIDS["$TASK_PID"]=1
done

mapfile -t TASK_BURN_GPU_PIDS < <(gpu_pids)
[[ "${#TASK_BURN_GPU_PIDS[@]}" -eq 8 ]] \
    || die "expected 8 GPU burn processes, found ${#TASK_BURN_GPU_PIDS[@]}"
for TASK_PID in "${TASK_BURN_GPU_PIDS[@]}"; do
    [[ -n "${TASK_EXPECTED_BURN_PIDS[$TASK_PID]:-}" ]] \
        || die "unexpected GPU process after burn launch: $TASK_PID"
    TASK_CMDLINE="$(tr '\0' ' ' < "/proc/$TASK_PID/cmdline")"
    [[ "$TASK_CMDLINE" == *scripts/gpu_burn.py* ]] \
        || die "GPU PID is not project burn: $TASK_PID $TASK_CMDLINE"
done

declare -A TASK_BURNS_PER_UUID=()
while IFS=',' read -r TASK_GPU_UUID TASK_PID; do
    TASK_GPU_UUID="${TASK_GPU_UUID//[[:space:]]/}"
    TASK_PID="${TASK_PID//[[:space:]]/}"
    [[ -n "$TASK_GPU_UUID" && -n "$TASK_PID" ]] || continue
    TASK_BURNS_PER_UUID["$TASK_GPU_UUID"]=$((
        ${TASK_BURNS_PER_UUID["$TASK_GPU_UUID"]:-0} + 1
    ))
done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits)
mapfile -t TASK_GPU_UUIDS < <(
    nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits \
        | sed 's/[[:space:]]//g'
)
[[ "${#TASK_GPU_UUIDS[@]}" -eq 8 ]] || die 'expected 8 GPU UUIDs'
for TASK_GPU_UUID in "${TASK_GPU_UUIDS[@]}"; do
    [[ "${TASK_BURNS_PER_UUID[$TASK_GPU_UUID]:-0}" -eq 1 ]] \
        || die "expected one burn on $TASK_GPU_UUID, found ${TASK_BURNS_PER_UUID[$TASK_GPU_UUID]:-0}"
done
for TASK_GPU in 0 1 2 3 4 5 6 7; do
    grep -Fq 'gpu_burn_ready' "/tmp/project_gpu_burn_nested_phase1_gpu${TASK_GPU}.log" \
        || die "burn readiness marker missing for GPU $TASK_GPU"
done
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,power.draw \
    --format=csv,noheader
echo 'TH2 NESTED PHASE-1 COMPLETE; GPU BURNS VERIFIED ON ALL 8 GPUS'
echo '=== supervising burns; this watcher intentionally remains active ==='
set +e
wait -n "${TASK_CHILD_PIDS[@]}"
TASK_FIRST_EXIT=$?
set -e
die "a burn worker exited with status $TASK_FIRST_EXIT"
