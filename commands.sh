#1 +60+a
#th2-verify-independent-lr-checkpoint-10000-complete-20260823
set -euo pipefail

TASK_OUTPUT=/mnt/local/_outputs/deep-llms_th2/lowrank_independent_output_r128
TASK_CHECKPOINT="$TASK_OUTPUT/checkpoint-10000"
TASK_LOG_DIR=/mnt/local/_outputs/deep-llms_th2/logs/lowrank_independent_output_r128
TASK_EXPERIMENT_LOG="$TASK_LOG_DIR/experiments.log"
TASK_TRAIN_LOG="$TASK_LOG_DIR/lowrank_independent_output_r128.log"
TASK_PYTHON=/mnt/local/conda-py311/envs/sparse_emb/bin/python3.11

echo '=== completion timestamp and GPU state ==='
date -u
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader

echo '=== ensure training and runner processes exited ==='
if pgrep -af '^/mnt/local/conda-py311/envs/sparse_emb/bin/python3[.]11 (-u run_experiments[.]py --experiments 12|-m accelerate[.]commands[.]launch train_compositional[.]py|-u train_compositional[.]py)'; then
    echo 'ERROR: training or runner process is still alive'
    exit 1
fi
mapfile -t TASK_GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -u
)
if [ "${#TASK_GPU_PIDS[@]}" -ne 0 ]; then
    echo "ERROR: unexpected GPU processes remain: ${TASK_GPU_PIDS[*]}"
    exit 1
fi
echo 'gpu_compute_processes=0'

echo '=== verify complete checkpoint-10000 ==='
test -d "$TASK_CHECKPOINT"
for required in config.json model.safetensors trainer_state.json optimizer.pt scheduler.pt \
                embedding.pt output_head.pt rng_state_0.pth rng_state_1.pth \
                rng_state_2.pth rng_state_3.pth rng_state_4.pth rng_state_5.pth \
                rng_state_6.pth rng_state_7.pth; do
    test -s "$TASK_CHECKPOINT/$required" || {
        echo "ERROR: missing or empty $TASK_CHECKPOINT/$required"
        exit 1
    }
done

"$TASK_PYTHON" - "$TASK_CHECKPOINT/trainer_state.json" <<'PY'
import json
import math
import sys

with open(sys.argv[1]) as handle:
    state = json.load(handle)
assert state["global_step"] == 10_000, state["global_step"]
metrics = [row for row in state.get("log_history", []) if "loss" in row]
assert len(metrics) == 1_000, len(metrics)
for row in metrics:
    for key in ("loss", "grad_norm", "learning_rate"):
        assert math.isfinite(float(row[key])), (key, row[key])
print("global_step=", state["global_step"], sep="")
print("finite_metric_rows=", len(metrics), sep="")
print("final_metrics=", metrics[-1], sep="")
PY

echo '=== verify runner completion ==='
test -s "$TASK_EXPERIMENT_LOG"
test -s "$TASK_TRAIN_LOG"
grep -F 'DONE: lowrank_independent_output_r128 — STOPPED at step 10000' "$TASK_EXPERIMENT_LOG"
grep -F 'All 1 experiments done' "$TASK_EXPERIMENT_LOG"
if grep -E -i 'Traceback|CUDA out of memory|OutOfMemoryError|RuntimeError:|NCCL[^[:cntrl:]]*(unhandled|system error|remote process exited|watchdog|collective operation timeout)' "$TASK_TRAIN_LOG"; then
    echo 'ERROR: fatal signature found in training log'
    exit 1
fi
tail -12 "$TASK_EXPERIMENT_LOG"
du -sh "$TASK_OUTPUT"
echo 'TH2 INDEPENDENT LR128 OUTPUT CHECKPOINT 10000 COMPLETE'
