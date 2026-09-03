#1 +60+a
#th2-readonly-progress-three-experiments-20260903-a08
set -euo pipefail

TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_LOG_DIR="$TASK_OUTPUT_BASE/logs/ranklift_hashedv2_btmos_10k_20260903_a01"
TASK_NAMES=(
    ranklift_tied_c124_m460
    product_code_quota_h6144
    btmos_k3_c256_lb
)

echo '=== time ==='
date -u

echo '=== sequential runner ==='
pgrep -af '[r]un_experiments.py.*--stop-at-step 10000' || echo 'runner=absent'
tail -40 "$TASK_LOG_DIR/experiments.log"

echo '=== current checkpoints ==='
for TASK_NAME in "${TASK_NAMES[@]}"; do
    TASK_OUTPUT="$TASK_OUTPUT_BASE/$TASK_NAME"
    if [ -d "$TASK_OUTPUT" ]; then
        TASK_LATEST=$(find "$TASK_OUTPUT" -mindepth 1 -maxdepth 1 -type d \
            -name 'checkpoint-*' -printf '%f\n' | sort -V | tail -1)
        echo "$TASK_NAME latest=${TASK_LATEST:-none}"
        if [ -n "$TASK_LATEST" ] && [ -s "$TASK_OUTPUT/$TASK_LATEST/trainer_state.json" ]; then
            /mnt/local/conda-py311/envs/sparse_emb/bin/python3.11 - \
                "$TASK_OUTPUT/$TASK_LATEST/trainer_state.json" <<'PY'
import json
import math
import sys
state = json.load(open(sys.argv[1]))
losses = [x['loss'] for x in state.get('log_history', []) if 'loss' in x]
print('global_step=', state.get('global_step'), 'finite_losses=', all(math.isfinite(x) for x in losses), 'recent_losses=', losses[-5:])
PY
        fi
    else
        echo "$TASK_NAME output=not_started"
    fi
done

echo '=== active experiment recent progress ==='
for TASK_NAME in "${TASK_NAMES[@]}"; do
    TASK_LOG="$TASK_LOG_DIR/$TASK_NAME.log"
    if [ -s "$TASK_LOG" ]; then
        echo "--- $TASK_NAME ---"
        tail -c 300000 "$TASK_LOG" | tr '\r' '\n' |
            grep -E '[0-9]+/33339|loss|Saving model checkpoint|Training completed' |
            tail -20 || true
    fi
done

echo '=== fatal signatures ==='
if grep -H -E -i \
        'Traceback|CUDA out of memory|OutOfMemoryError|ChildFailedError|ProcessExitedException|NCCL[^[:cntrl:]]*(unhandled|system error|remote process exited|watchdog|collective operation timeout)|Segmentation fault|Bus error|nan loss|inf loss' \
        "$TASK_LOG_DIR"/*.log; then
    echo 'FATAL_SIGNATURES_FOUND=YES'
else
    echo 'FATAL_SIGNATURES_FOUND=NO'
fi

echo '=== GPU state ==='
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory \
    --format=csv,noheader

TASK_MARKER="$TASK_OUTPUT_BASE/status/ranklift_hashedv2_btmos_10k_20260903_a01.complete"
if [ -s "$TASK_MARKER" ]; then
    echo '=== completion marker ==='
    cat "$TASK_MARKER"
else
    echo 'completion_marker=absent'
fi
echo 'TH2 READ-ONLY TRAINING PROGRESS SNAPSHOT COMPLETE'
