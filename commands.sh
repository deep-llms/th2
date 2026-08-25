#1 +60+a
#th2-check-matched-tied-controls-final-completion-20260825
set -euo pipefail

TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_LOG_DIR="$TASK_OUTPUT_BASE/logs/b200_matched_global_lr_then_dense_tied_20260824"
TASK_GLOBAL_OUTPUT="$TASK_OUTPUT_BASE/global_lowrank_tied_r128_b200"
TASK_DENSE_OUTPUT="$TASK_OUTPUT_BASE/dense_tied_baseline_b200"

echo '=== host and GPU state (read only) ==='
date -u
hostname
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
mapfile -t TASK_GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -u
)
echo "gpu_compute_process_count=${#TASK_GPU_PIDS[@]}"
for pid in "${TASK_GPU_PIDS[@]}"; do
    if [ -r "/proc/$pid/cmdline" ]; then
        printf 'gpu_pid=%s cmd=' "$pid"
        tr '\0' ' ' < "/proc/$pid/cmdline"
        echo
    fi
done

echo '=== matched runner and training processes (read only) ==='
TASK_MATCHED_PROCESS_COUNT=0
for proc in /proc/[0-9]*; do
    [ -r "$proc/cmdline" ] || continue
    cmd=$(tr '\0' ' ' < "$proc/cmdline" 2>/dev/null || true)
    if [[ "$cmd" == *python* ]] && {
        [[ "$cmd" == *"run_experiments.py --experiments 13 14"* ]] \
            || [[ "$cmd" == *"global_lowrank_tied_r128_b200"* ]] \
            || [[ "$cmd" == *"dense_tied_baseline_b200"* ]];
    }; then
        TASK_MATCHED_PROCESS_COUNT=$((TASK_MATCHED_PROCESS_COUNT + 1))
        printf 'matched_pid=%s cmd=' "${proc#/proc/}"
        printf '%s\n' "$cmd"
    fi
done
echo "matched_process_count=$TASK_MATCHED_PROCESS_COUNT"

echo '=== checkpoint artifact validation ==='
python3 - "$TASK_GLOBAL_OUTPUT" "$TASK_DENSE_OUTPUT" <<'PY'
import json
import os
import sys

experiments = (
    (
        "global_lowrank_tied_r128_b200",
        sys.argv[1],
        [
            "config.json", "model.safetensors", "trainer_state.json",
            "optimizer.pt", "scheduler.pt", "embedding.pt",
            *(f"rng_state_{rank}.pth" for rank in range(8)),
        ],
    ),
    (
        "dense_tied_baseline_b200",
        sys.argv[2],
        [
            "config.json", "model.safetensors", "trainer_state.json",
            "optimizer.pt", "scheduler.pt",
            *(f"rng_state_{rank}.pth" for rank in range(8)),
        ],
    ),
)

for name, output_dir, required in experiments:
    steps = []
    if os.path.isdir(output_dir):
        for entry in os.listdir(output_dir):
            if entry.startswith("checkpoint-"):
                try:
                    steps.append(int(entry.removeprefix("checkpoint-")))
                except ValueError:
                    pass
    latest = max(steps, default=-1)
    target = os.path.join(output_dir, "checkpoint-10000")
    missing = [
        filename for filename in required
        if not os.path.isfile(os.path.join(target, filename))
        or os.path.getsize(os.path.join(target, filename)) == 0
    ]
    saved_step = None
    state_path = os.path.join(target, "trainer_state.json")
    if os.path.isfile(state_path):
        try:
            with open(state_path) as handle:
                saved_step = json.load(handle).get("global_step")
        except (OSError, ValueError):
            saved_step = "INVALID"
    complete = not missing and saved_step == 10000
    print(
        f"experiment={name} latest_checkpoint={latest} "
        f"checkpoint_10000_complete={str(complete).lower()} "
        f"trainer_global_step={saved_step} missing={missing}"
    )
PY

echo '=== sequential runner summary ==='
if [ -s "$TASK_LOG_DIR/experiments.log" ]; then
    tail -50 "$TASK_LOG_DIR/experiments.log"
else
    echo "missing_or_empty=$TASK_LOG_DIR/experiments.log"
fi

echo '=== training log tails and fatal-signature scan ==='
for name in global_lowrank_tied_r128_b200 dense_tied_baseline_b200; do
    log="$TASK_LOG_DIR/$name.log"
    echo "--- $name ---"
    if [ -s "$log" ]; then
        tail -30 "$log"
        if grep -Eqi 'Traceback|NCCL.*(error|fatal)|CUDA out of memory|OutOfMemoryError|RuntimeError|FAILED|Error:' "$log"; then
            echo "fatal_signature_detected=$name"
        else
            echo "fatal_signature_detected=none"
        fi
    else
        echo "missing_or_empty=$log"
    fi
done

echo 'TH2 MATCHED TIED CONTROL FINAL COMPLETION CHECK FINISHED'
