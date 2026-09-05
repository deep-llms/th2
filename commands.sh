#1 +180+a
#th2-readonly-status-tiered-c512-lb-groupreduce-20260905-a03
set -euo pipefail

TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_TIERED="$TASK_OUTPUT_BASE/tiered_ranklift_lb_t4_c512"
TASK_CONTROL="$TASK_OUTPUT_BASE/groupreduce_matched_lb_t4"
TASK_LOG_DIR="$TASK_OUTPUT_BASE/logs/tiered_c512_lb_groupreduce_10k_20260904_a01"
TASK_MARKER="$TASK_OUTPUT_BASE/status/tiered_c512_lb_groupreduce_10k_20260904_a01.complete"

echo '=== read-only identity and GPU snapshot ==='
date -u
hostname
nvidia-smi \
    --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader
nvidia-smi \
    --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits || true

echo '=== read-only relevant processes ==='
pgrep -af '[r]un_experiments.py|[a]ccelerate.commands.launch|[t]rain_compositional.py|[l]lm_pretrain_burn.py' || true

echo '=== completion marker ==='
if [[ -s "$TASK_MARKER" ]]; then
    cat "$TASK_MARKER"
else
    echo 'COMPLETION_MARKER_NOT_PRESENT'
fi

echo '=== output and checkpoint state ==='
for TASK_OUTPUT in "$TASK_TIERED" "$TASK_CONTROL"; do
    echo "OUTPUT=$TASK_OUTPUT"
    if [[ ! -d "$TASK_OUTPUT" ]]; then
        echo 'NOT_CREATED'
        continue
    fi
    find "$TASK_OUTPUT" -mindepth 1 -maxdepth 1 -type d \
        -name 'checkpoint-*' -printf '%f\n' | sort -V | tail -10 || true
    if [[ -s "$TASK_OUTPUT/train_progress.csv" ]]; then
        tail -5 "$TASK_OUTPUT/train_progress.csv"
    fi
done

echo '=== validate all readable checkpoint states and finite losses ==='
/mnt/local/conda-py311/envs/sparse_emb/bin/python3.11 - \
    "$TASK_TIERED" "$TASK_CONTROL" "$TASK_LOG_DIR" <<'PY'
import json
import math
import pathlib
import re
import sys

outputs = [pathlib.Path(value) for value in sys.argv[1:3]]
log_dir = pathlib.Path(sys.argv[3])
fatal_markers = (
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "OutOfMemoryError",
    "ChildFailedError",
    "ProcessExitedException",
    "Segmentation fault",
    "Bus error",
)

for output in outputs:
    valid_steps = []
    for checkpoint in output.glob("checkpoint-*") if output.is_dir() else ():
        try:
            step = int(checkpoint.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        state_path = checkpoint / "trainer_state.json"
        if not state_path.is_file() or not state_path.stat().st_size:
            continue
        state = json.loads(state_path.read_text())
        assert int(state["global_step"]) == step, (checkpoint, state["global_step"])
        recorded = [
            float(row["loss"])
            for row in state.get("log_history", [])
            if "loss" in row
        ]
        assert all(math.isfinite(value) for value in recorded), checkpoint
        valid_steps.append(step)
    print(output.name, "VALID_STEPS", sorted(valid_steps)[-10:])

    log_path = log_dir / f"{output.name}.log"
    if not log_path.is_file():
        print(output.name, "LOG_NOT_CREATED")
        continue
    text = log_path.read_text(errors="replace")
    fatal = [marker for marker in fatal_markers if marker in text]
    print(output.name, "FATAL_MARKERS", fatal)
    assert not fatal, (output.name, fatal)
    losses = [
        float(value)
        for value in re.findall(r"'loss': ['\"]?([0-9.eE+-]+)", text)
    ]
    assert all(math.isfinite(value) for value in losses), output.name
    if losses:
        print(
            output.name,
            "FINITE_LOSSES",
            f"count={len(losses)}",
            f"first={losses[0]:.6f}",
            f"last={losses[-1]:.6f}",
            f"minimum={min(losses):.6f}",
        )
PY

echo '=== runner summary and experiment log tails ==='
if [[ -s "$TASK_LOG_DIR/experiments.log" ]]; then
    cat "$TASK_LOG_DIR/experiments.log"
else
    echo 'EXPERIMENTS_LOG_MISSING'
fi
for TASK_NAME in tiered_ranklift_lb_t4_c512 groupreduce_matched_lb_t4; do
    TASK_LOG="$TASK_LOG_DIR/$TASK_NAME.log"
    echo "LOG=$TASK_LOG"
    if [[ -s "$TASK_LOG" ]]; then
        tail -60 "$TASK_LOG"
    else
        echo 'NOT_CREATED'
    fi
done

echo 'TH2 TIERED-C512/LB-GROUPREDUCE READ-ONLY STATUS CHECK COMPLETE'
