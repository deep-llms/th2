#1 +30+a
#th2-cancel-four-model-eval-finetune-workflow-20260830-a01
set -euo pipefail

TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_RUNTIME=/tmp/run_phase1_four_model_eval_finetune_burn_6a8d9c6825a9.sh
TASK_FINETUNE_OUTPUT="$TASK_OUTPUT_BASE/finetune_nested_groupreduce_two_dense_10k_20260830"
TASK_CHECKPOINTS=(
    "$TASK_OUTPUT_BASE/nested_ladder_tied_t4/checkpoint-10000"
    "$TASK_OUTPUT_BASE/groupreduce_matched_nested_tied_t4/checkpoint-10000"
    "$TASK_OUTPUT_BASE/dense_tied_baseline_b200/checkpoint-10000"
    "$TASK_OUTPUT_BASE/dense_tied_baseline_b200_ddp_default/checkpoint-10000"
)

collect_targets() {
    python3 - "$TASK_RUNTIME" "$TASK_FINETUNE_OUTPUT" "${TASK_CHECKPOINTS[@]}" <<'PY'
import os
import sys

runtime, finetune_output, *checkpoints = sys.argv[1:]


def read_args(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            raw = handle.read()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return []
    return [part.decode(errors="replace") for part in raw.rstrip(b"\0").split(b"\0") if part]


def has_script(args, suffix):
    return any(arg == suffix or arg.endswith("/" + suffix) for arg in args[1:])


skip = {1, os.getpid(), os.getppid()}
for entry in os.scandir("/proc"):
    if not entry.name.isdigit():
        continue
    pid = int(entry.name)
    if pid in skip:
        continue
    args = read_args(pid)
    if not args:
        continue

    match = False
    executable = os.path.basename(args[0])
    if executable in {"bash", "sh"} and len(args) >= 2 and args[1] == runtime:
        match = True
    elif has_script(args, "eval/eval_parallel.py") and all(
        checkpoint in args for checkpoint in checkpoints
    ):
        match = True
    elif has_script(args, "eval/eval_checkpoint.py") and any(
        checkpoint in args for checkpoint in checkpoints
    ):
        match = True
    elif has_script(args, "finetune/run_all.py") and finetune_output in args:
        match = True
    elif (
        has_script(args, "finetune/train.py")
        and finetune_output in args
        and any(checkpoint in args for checkpoint in checkpoints)
    ):
        match = True

    if match:
        print(pid)
PY
}

show_process() {
    local pid="$1"
    local cmdline
    cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
    echo "target_pid=$pid cmdline=$cmdline"
}

echo '=== identify only this four-model workflow and its eval/finetune jobs ==='
date -u
mapfile -t TASK_TARGETS < <(collect_targets | sort -nu)
if [[ "${#TASK_TARGETS[@]}" -eq 0 ]]; then
    echo 'No matching workflow, eval, or finetune process is currently running.'
else
    for TASK_PID in "${TASK_TARGETS[@]}"; do
        [[ "$TASK_PID" -ne 1 ]] || { echo 'ERROR: refusing to signal PID 1' >&2; exit 1; }
        show_process "$TASK_PID"
    done

    echo "Sending TERM to exact targets: ${TASK_TARGETS[*]}"
    kill -TERM "${TASK_TARGETS[@]}" 2>/dev/null || true

    for TASK_ATTEMPT in $(seq 1 30); do
        mapfile -t TASK_REMAINING < <(collect_targets | sort -nu)
        [[ "${#TASK_REMAINING[@]}" -eq 0 ]] && break
        sleep 1
    done

    if [[ "${#TASK_REMAINING[@]}" -gt 0 ]]; then
        echo "TERM timeout; sending KILL to still-matching targets: ${TASK_REMAINING[*]}"
        for TASK_PID in "${TASK_REMAINING[@]}"; do
            [[ "$TASK_PID" -ne 1 ]] || { echo 'ERROR: refusing to signal PID 1' >&2; exit 1; }
            show_process "$TASK_PID"
        done
        kill -KILL "${TASK_REMAINING[@]}" 2>/dev/null || true
    fi
fi

sleep 3
mapfile -t TASK_FINAL < <(collect_targets | sort -nu)
if [[ "${#TASK_FINAL[@]}" -ne 0 ]]; then
    echo "ERROR: matching workflow/eval/finetune processes remain: ${TASK_FINAL[*]}" >&2
    for TASK_PID in "${TASK_FINAL[@]}"; do
        show_process "$TASK_PID"
    done
    exit 1
fi
echo 'CANCEL VERIFIED: no matching workflow, eval, or finetune process remains.'

echo '=== GPU state after cancellation (read-only; do not stop existing burns) ==='
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,power.draw \
    --format=csv,noheader
mapfile -t TASK_GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -nu
)
if [[ "${#TASK_GPU_PIDS[@]}" -eq 0 ]]; then
    echo 'ALL GPUS FREE AFTER CANCELLATION'
else
    echo "GPU processes remain (not signalled): ${TASK_GPU_PIDS[*]}"
    for TASK_PID in "${TASK_GPU_PIDS[@]}"; do
        show_process "$TASK_PID"
    done
fi

echo '=== preserve and report any partial artifacts; delete nothing ==='
for TASK_CHECKPOINT in "${TASK_CHECKPOINTS[@]}"; do
    echo "checkpoint=$TASK_CHECKPOINT"
    find "$TASK_CHECKPOINT" -maxdepth 1 -type f \
        \( -name 'eval.log' -o -name 'eval_ppl.json' -o -name 'eval_benchmarks.json' \) \
        -printf '  %f %s bytes\n' 2>/dev/null | sort || true
done
if [[ -e "$TASK_FINETUNE_OUTPUT" ]]; then
    echo "finetune_output_exists=$TASK_FINETUNE_OUTPUT"
    find "$TASK_FINETUNE_OUTPUT" -maxdepth 1 -type f -printf '  %f %s bytes\n' \
        2>/dev/null | sort | head -100 || true
else
    echo "finetune_output_absent=$TASK_FINETUNE_OUTPUT"
fi
echo 'TH2 FOUR-MODEL EVAL/FINETUNE CANCELLATION COMPLETE'
