#1 +60+a
#th2-readonly-check-final-rse-hashed-ready-for-eval-20260901-a04
set -euo pipefail

TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_RUN_LOG="$TASK_OUTPUT_BASE/logs/final_rse_b8a8_hashed_10k_20260831_a02/run_experiments.log"
TASK_PYTHON=/mnt/local/conda-py311/envs/sparse_emb/bin/python3.11
TASK_PROJECT_DIR=/mnt/local/@PROJECT@

cd "$TASK_PROJECT_DIR"

echo '=== identity and GPU state ==='
date -u
hostname
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader

echo '=== GPU compute owners ==='
mapfile -t TASK_GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | sed 's/^[[:space:]]*//;s/[[:space:]]*$//;/^$/d' | sort -u
)
echo "unique_gpu_pid_count=${#TASK_GPU_PIDS[@]}"
for TASK_PID in "${TASK_GPU_PIDS[@]}"; do
    case "$TASK_PID" in
        *[!0-9]*|'') echo "ERROR: invalid GPU PID: $TASK_PID"; exit 1 ;;
    esac
    TASK_CMD=$(tr '\0' ' ' < "/proc/$TASK_PID/cmdline")
    echo "pid=$TASK_PID ppid=$(awk '/^PPid:/{print $2}' "/proc/$TASK_PID/status") cmd=$TASK_CMD"
done

echo '=== relevant tmux sessions ==='
tmux list-sessions 2>/dev/null | grep -E 'final-rse|hashed|burn|eval|finetune' || true

echo '=== checkpoint integrity ==='
for TASK_EXPERIMENT in \
    residual_subspace_experts_tied_g12_r120_q80 \
    product_code_hashed_h2048; do
    TASK_CHECKPOINT="$TASK_OUTPUT_BASE/$TASK_EXPERIMENT/checkpoint-10000"
    echo "experiment=$TASK_EXPERIMENT"
    test -d "$TASK_CHECKPOINT" || {
        echo "checkpoint_10000=ABSENT"
        continue
    }
    TASK_MISSING=0
    for TASK_FILE in config.json model.safetensors trainer_state.json \
                     optimizer.pt scheduler.pt embedding.pt \
                     rng_state_0.pth rng_state_1.pth rng_state_2.pth \
                     rng_state_3.pth rng_state_4.pth rng_state_5.pth \
                     rng_state_6.pth rng_state_7.pth; do
        if [ ! -s "$TASK_CHECKPOINT/$TASK_FILE" ]; then
            echo "missing_or_empty=$TASK_FILE"
            TASK_MISSING=1
        fi
    done
    "$TASK_PYTHON" - "$TASK_CHECKPOINT/trainer_state.json" <<'PY'
import json
import math
import sys

with open(sys.argv[1]) as handle:
    state = json.load(handle)
assert state["global_step"] == 10000, state["global_step"]
rows = [row for row in state.get("log_history", []) if "loss" in row]
assert rows, "no loss rows"
for row in rows:
    for key in ("loss", "grad_norm", "learning_rate"):
        assert math.isfinite(float(row[key])), (key, row[key])
print(f"global_step={state['global_step']} finite_loss_rows={len(rows)}")
print(f"last_loss_row={rows[-1]}")
PY
    test "$TASK_MISSING" -eq 0
    echo 'checkpoint_10000=COMPLETE'
done

echo '=== sequential runner tail ==='
if [ -s "$TASK_RUN_LOG" ]; then
    tail -n 100 "$TASK_RUN_LOG"
else
    echo "run_log_absent=$TASK_RUN_LOG"
fi

echo 'TH2 READONLY RSE HASHED EVAL READINESS CHECK COMPLETE'
