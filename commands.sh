#1 +30+a
#th2-check-nested-phase1-completion-and-burns-20260830-a01
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
pgrep -af '[t]rain_compositional.py' | head -20 || echo 'training_process=absent'
pgrep -af '[a]ccelerate.commands.launch|[a]ccelerate launch' \
    || echo 'accelerate_launcher=absent'
pgrep -af '[g]pu_burn.py' || echo 'gpu_burn_process=absent'

echo '=== relevant tmux sessions ==='
tmux list-panes -a -F 'session=#{session_name} pane_pid=#{pane_pid} command=#{pane_current_command} dead=#{pane_dead} status=#{pane_dead_status}' 2>&1 \
    | grep -E 'nested|groupreduce|burn' || true

echo '=== sequential experiment log ==='
if [ -s "$TASK_EXPERIMENT_LOG" ]; then
    cat "$TASK_EXPERIMENT_LOG"
else
    echo 'experiment_log=missing_or_empty'
fi

echo '=== output validation ==='
for TASK_NAME in "${TASK_NAMES[@]}"; do
    TASK_OUTPUT="$TASK_OUTPUT_BASE/$TASK_NAME"
    TASK_CHECKPOINT="$TASK_OUTPUT/checkpoint-10000"
    TASK_TRAIN_LOG="$TASK_LOG_DIR/$TASK_NAME.log"
    echo "--- $TASK_NAME ---"
    if [ -d "$TASK_OUTPUT" ]; then
        du -sh "$TASK_OUTPUT"
        find "$TASK_OUTPUT" -mindepth 1 -maxdepth 1 -type d \
            -name 'checkpoint-*' -printf '%f\n' | sort -V | tail -10
    else
        echo 'output=absent'
    fi
    if [ -s "$TASK_CHECKPOINT/trainer_state.json" ] \
            && [ -s "$TASK_CHECKPOINT/config.json" ]; then
        "$TASK_PYTHON" - "$TASK_NAME" "$TASK_CHECKPOINT" <<'PY'
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
    f"checkpoint={os.path.basename(checkpoint)}",
    f"global_step={int(state['global_step'])}",
    f"arm={comp.get('arm')}",
    f"tie_output={comp.get('tie_output')}",
)
assert int(state["global_step"]) == 10000
assert comp.get("tie_output") is True
if name == "nested_ladder_tied_t4":
    assert comp.get("arm") == "nested_ladder", comp
    assert comp.get("nested_tier_ranks") == "64,128,320,512", comp
    assert comp.get("nested_tier_populations") == "151936,32768,8192,2048", comp
else:
    assert comp.get("arm") == "groupreduce", comp
    assert comp.get("groupreduce_ranks") == "1024,512,192,64", comp
    assert comp.get("groupreduce_populations") == "2048,6144,24576,119168", comp
print("CHECKPOINT_10000_ARCHITECTURE_OK")
PY
    else
        echo 'checkpoint_10000=not_complete'
    fi
    if [ -s "$TASK_TRAIN_LOG" ]; then
        echo "training_log_bytes=$(stat -c %s "$TASK_TRAIN_LOG")"
        if grep -HniE 'CUDA out of memory|OutOfMemoryError|NCCL.*(unhandled|system error|remote process exited|watchdog|timeout)|Segmentation fault|Bus error' \
                "$TASK_TRAIN_LOG"; then
            TASK_FATAL=1
        fi
        tail -c 180000 "$TASK_TRAIN_LOG" | tr '\r' '\n' \
            | grep -E "\{'loss':" | tail -10 || true
    else
        echo 'training_log=absent'
    fi
done

echo '=== watcher pane tail ==='
TASK_WATCHER_SESSION="$(
    tmux list-sessions -F '#{session_name}' 2>/dev/null \
        | grep 'watch-nested-phase1-then-supervise-burns' | head -1
)"
if [ -n "$TASK_WATCHER_SESSION" ]; then
    echo "watcher_session=$TASK_WATCHER_SESSION"
    tmux capture-pane -p -t "$TASK_WATCHER_SESSION" -S -120 2>&1 | tail -120
else
    echo 'watcher_session=absent'
fi

echo '=== GPU state ==='
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
    echo 'ERROR: fatal signature found in training logs' >&2
    exit 1
fi
echo 'TH2 NESTED PHASE1 COMPLETION CHECK FINISHED'
