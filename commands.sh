#1 +120+a
#th2-readonly-current-tiered-control-status-20260905-a05
set -euo pipefail

TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_LOG_DIR="$TASK_OUTPUT_BASE/logs/tiered_c512_lb_groupreduce_10k_20260904_a01"
TASK_MARKER="$TASK_OUTPUT_BASE/status/tiered_c512_lb_groupreduce_10k_20260904_a01.complete"

echo '=== read-only current state ==='
date -u
nvidia-smi \
    --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader
nvidia-smi \
    --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits || true

echo '=== active workload identities ==='
mapfile -t TASK_GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits |
        sed 's/^[[:space:]]*//;s/[[:space:]]*$//;/^$/d' | sort -nu
)
for TASK_PID in "${TASK_GPU_PIDS[@]}"; do
    [[ "$TASK_PID" =~ ^[0-9]+$ && "$TASK_PID" -ne 1 ]] || continue
    printf 'pid=%s cmd=' "$TASK_PID"
    tr '\0' ' ' < "/proc/$TASK_PID/cmdline"
    printf '\n'
done

echo '=== completion marker ==='
if [[ -s "$TASK_MARKER" ]]; then
    cat "$TASK_MARKER"
else
    echo 'COMPLETION_MARKER_NOT_PRESENT'
fi

echo '=== latest checkpoint states ==='
/mnt/local/conda-py311/envs/sparse_emb/bin/python3.11 - \
    "$TASK_OUTPUT_BASE/tiered_ranklift_lb_t4_c512" \
    "$TASK_OUTPUT_BASE/groupreduce_matched_lb_t4" <<'PY'
import json
import pathlib
import sys

for raw in sys.argv[1:]:
    output = pathlib.Path(raw)
    completed = []
    for checkpoint in output.glob("checkpoint-*") if output.is_dir() else ():
        try:
            step = int(checkpoint.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        state_path = checkpoint / "trainer_state.json"
        if state_path.is_file() and state_path.stat().st_size:
            state = json.loads(state_path.read_text())
            if int(state["global_step"]) == step:
                completed.append(step)
    print(output.name, "latest_valid_checkpoint", max(completed, default=None))
PY

echo '=== runner state ==='
tail -30 "$TASK_LOG_DIR/experiments.log"
echo '=== active control log tail ==='
tail -40 "$TASK_LOG_DIR/groupreduce_matched_lb_t4.log" 2>/dev/null || true
echo 'TH2 CURRENT TRAINING READ-ONLY STATUS CHECK COMPLETE'
