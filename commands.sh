#1 +60+a
#th2-readonly-current-tiered-control-status-20260905-a07
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
    TASK_PARENT=$(awk '/^PPid:/ {print $2}' "/proc/$TASK_PID/status")
    if [[ "$TASK_PARENT" -gt 1 && -r "/proc/$TASK_PARENT/cmdline" ]]; then
        printf 'parent=%s cmd=' "$TASK_PARENT"
        tr '\0' ' ' < "/proc/$TASK_PARENT/cmdline"
        printf '\n'
    fi
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
import math
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
    checkpoint = output / 'checkpoint-10000'
    if (checkpoint / 'trainer_state.json').is_file():
        required = ['config.json', 'model.safetensors', 'trainer_state.json',
                    'optimizer.pt', 'scheduler.pt', 'embedding.pt']
        required += [f'rng_state_{i}.pth' for i in range(8)]
        missing = [n for n in required if not (checkpoint/n).is_file() or not (checkpoint/n).stat().st_size]
        state = json.loads((checkpoint/'trainer_state.json').read_text())
        rows = [r for r in state.get('log_history', []) if 'loss' in r]
        finite = bool(rows) and all(math.isfinite(float(r[k])) for r in rows for k in ('loss','grad_norm','learning_rate') if k in r)
        print('CHECKPOINT_10000', output.name, 'step', state['global_step'], 'missing', missing, 'finite_metrics', finite, 'last', rows[-1] if rows else None)
PY

echo '=== runner state ==='
tail -30 "$TASK_LOG_DIR/experiments.log"
echo '=== active control log tail ==='
tail -40 "$TASK_LOG_DIR/groupreduce_matched_lb_t4.log" 2>/dev/null || true
echo '=== handoff log tail ==='
ls -lah "$TASK_LOG_DIR"
find "$TASK_OUTPUT_BASE" -maxdepth 3 -type f -iname '*burn*' -printf '%p %s bytes %TY-%Tm-%Td %TH:%TM\n'
echo 'TH2 CURRENT TRAINING READ-ONLY STATUS CHECK COMPLETE'
