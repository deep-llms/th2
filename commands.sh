#1 +30+a
#th2-check-nested-ladder-phase1-current-run-20260829-a02
set -uo pipefail

TASK_PYTHON=/mnt/local/conda-py311/envs/sparse_emb/bin/python3.11
TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_LOG_DIR="$TASK_OUTPUT_BASE/logs/nested_ladder_phase1_20260829"
TASK_EXPERIMENT_LOG="$TASK_LOG_DIR/experiments.log"
TASK_NAMES=(nested_ladder_tied_t4 groupreduce_matched_nested_tied_t4)
TASK_FATAL=0

echo '=== identity ==='
date -u
hostname

echo '=== active workflow processes ==='
pgrep -af '[r]un_experiments.py' || echo 'sequential_runner=absent'
pgrep -af '[a]ccelerate.commands.launch|[a]ccelerate launch' \
    || echo 'accelerate_launcher=absent'
mapfile -t TASK_TRAIN_PIDS < <(pgrep -f '[t]rain_compositional.py' | sort -nu)
echo "train_related_process_count=${#TASK_TRAIN_PIDS[@]}"

echo '=== sequential runner log ==='
if [ -s "$TASK_EXPERIMENT_LOG" ]; then
    cat "$TASK_EXPERIMENT_LOG"
else
    echo 'experiment_log=missing_or_empty'
fi

echo '=== experiment outputs and training logs ==='
for TASK_NAME in "${TASK_NAMES[@]}"; do
    TASK_OUTPUT="$TASK_OUTPUT_BASE/$TASK_NAME"
    TASK_TRAIN_LOG="$TASK_LOG_DIR/$TASK_NAME.log"
    echo "--- $TASK_NAME ---"
    if [ -d "$TASK_OUTPUT" ]; then
        du -sh "$TASK_OUTPUT"
        find "$TASK_OUTPUT" -mindepth 1 -maxdepth 1 -type d \
            -name 'checkpoint-*' -printf '%f\n' | sort -V | tail -10
        TASK_LATEST_CHECKPOINT="$(
            find "$TASK_OUTPUT" -mindepth 1 -maxdepth 1 -type d \
                -name 'checkpoint-*' -printf '%p\n' | sort -V | tail -1
        )"
        if [ -n "$TASK_LATEST_CHECKPOINT" ] \
                && [ -s "$TASK_LATEST_CHECKPOINT/trainer_state.json" ] \
                && [ -s "$TASK_LATEST_CHECKPOINT/config.json" ]; then
            "$TASK_PYTHON" - "$TASK_NAME" "$TASK_LATEST_CHECKPOINT" <<'PY'
import json
import os
import sys

name, checkpoint = sys.argv[1:]
with open(os.path.join(checkpoint, "trainer_state.json"), encoding="utf-8") as handle:
    state = json.load(handle)
with open(os.path.join(checkpoint, "config.json"), encoding="utf-8") as handle:
    config = json.load(handle)
comp = config.get("compositional", {})
print(
    f"latest_checkpoint={os.path.basename(checkpoint)}",
    f"global_step={int(state['global_step'])}",
    f"arm={comp.get('arm')}",
    f"tie_output={comp.get('tie_output')}",
    f"ddp_expected_false={name in {'nested_ladder_tied_t4', 'groupreduce_matched_nested_tied_t4'}}",
)
assert int(state["global_step"]) > 0
assert comp.get("tie_output") is True
if name == "nested_ladder_tied_t4":
    assert comp.get("arm") == "nested_ladder", comp
    assert comp.get("nested_tier_ranks") == "64,128,320,512", comp
    assert comp.get("nested_tier_populations") == "151936,32768,8192,2048", comp
else:
    assert comp.get("arm") == "groupreduce", comp
    assert comp.get("groupreduce_ranks") == "1024,512,192,64", comp
    assert comp.get("groupreduce_populations") == "2048,6144,24576,119168", comp
print("CHECKPOINT_ARCHITECTURE_OK")
PY
        else
            echo 'complete_checkpoint=not_available_yet'
        fi
    else
        echo 'output=not_created_yet'
    fi

    if [ -s "$TASK_TRAIN_LOG" ]; then
        echo "training_log_bytes=$(stat -c %s "$TASK_TRAIN_LOG")"
        echo "training_log_mtime=$(stat -c %y "$TASK_TRAIN_LOG")"
        if grep -HniE 'CUDA out of memory|OutOfMemoryError|NCCL.*(unhandled|system error|remote process exited|watchdog|timeout)|Segmentation fault|Bus error' \
                "$TASK_TRAIN_LOG"; then
            TASK_FATAL=1
        fi
        tail -c 250000 "$TASK_TRAIN_LOG" | tr '\r' '\n' \
            | grep -E "Embedding:|Total parameters:|Trainable parameters:|\{'loss':" \
            | tail -25 || true
        echo 'raw_training_log_tail:'
        tail -c 120000 "$TASK_TRAIN_LOG" | tr '\r' '\n' | tail -100
    else
        echo 'training_log=not_created_yet'
    fi
done

echo '=== live GPU state ==='
mapfile -t TASK_GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -nu
)
echo "gpu_compute_pid_count=${#TASK_GPU_PIDS[@]}"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,power.draw \
    --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits || true

if [ "$TASK_FATAL" -ne 0 ]; then
    echo 'ERROR: fatal training signature detected' >&2
    exit 1
fi
echo 'TH2 NESTED LADDER PHASE1 CURRENT RUN CHECK COMPLETE'
