#1 +120+a
#th2-readonly-audit-three-experiment-training-20260903-a03
set -euo pipefail

TASK_PROJECT=/mnt/local/@PROJECT@
TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_LOG_DIR="$TASK_OUTPUT_BASE/logs/ranklift_hashedv2_btmos_10k_20260903_a01"
TASK_EXPERIMENT_LOG="$TASK_LOG_DIR/experiments.log"
TASK_COMPLETION_FILE="$TASK_OUTPUT_BASE/status/ranklift_hashedv2_btmos_10k_20260903_a01.complete"
TASK_ACCELERATE_TARGET=/mnt/local/.cache/huggingface/accelerate/default_config.yaml
TASK_PYTHON=/mnt/local/conda-py311/envs/sparse_emb/bin/python3.11
TASK_NAMES=(
    ranklift_tied_c124_m460
    product_code_quota_h6144
    btmos_k3_c256_lb
)

cd "$TASK_PROJECT"
echo '=== identity and deployed revision ==='
date -u
hostname
git rev-parse HEAD

echo '=== immutable inputs and Accelerate config ==='
echo '39b15eab8cf213d563dcf5137bb982e836bb8e3beba8e7def8dcddf21fe43594  resources/token_importance_langbalanced.npz' | sha256sum -c -
echo 'f43d19925f5add96c56913eccf57f3989d6cd52e69da761d879e22f901010ea5  resources/token_importance_quota.npz' | sha256sum -c -
echo '923db7f20a2df3d051180f67f9bea1f30c84c804651e313fa9961a9fd17a57e5  resources/accelerate_config.yaml' | sha256sum -c -
cmp resources/accelerate_config.yaml "$TASK_ACCELERATE_TARGET"
grep -E 'distributed_type:|mixed_precision:|num_processes:' "$TASK_ACCELERATE_TARGET"

echo '=== exact live workflow processes ==='
mapfile -t TASK_RUNNER_PIDS < <(pgrep -f '[r]un_experiments.py.*--stop-at-step 10000' | sort -nu)
echo "runner_count=${#TASK_RUNNER_PIDS[@]}"
for TASK_PID in "${TASK_RUNNER_PIDS[@]}"; do
    ps -p "$TASK_PID" -o pid=,ppid=,etimes=,stat=,args=
done
pgrep -af '[a]ccelerate.commands.launch|[a]ccelerate launch' || echo 'accelerate_launcher=absent'
mapfile -t TASK_TRAIN_PIDS < <(pgrep -f '[t]rain_compositional.py' | sort -nu)
echo "train_process_count=${#TASK_TRAIN_PIDS[@]}"
for TASK_PID in "${TASK_TRAIN_PIDS[@]}"; do
    ps -p "$TASK_PID" -o pid=,ppid=,etimes=,stat=,args=
done

echo '=== sequential runner state ==='
if [ -s "$TASK_EXPERIMENT_LOG" ]; then
    tail -80 "$TASK_EXPERIMENT_LOG"
else
    echo 'experiments_log=missing_or_empty'
fi
if [ -s "$TASK_COMPLETION_FILE" ]; then
    echo 'completion_marker=present'
    cat "$TASK_COMPLETION_FILE"
else
    echo 'completion_marker=absent_training_not_fully_verified_yet'
fi

echo '=== output/checkpoint/trainer-state inventory ==='
"$TASK_PYTHON" - "${TASK_NAMES[@]}" <<'PY'
import json
import math
import sys
from pathlib import Path

base = Path('/mnt/local/_outputs/@PROJECT@')
for name in sys.argv[1:]:
    out = base / name
    checkpoints = sorted(
        (p for p in out.glob('checkpoint-*') if p.is_dir()),
        key=lambda p: int(p.name.rsplit('-', 1)[1]),
    ) if out.is_dir() else []
    print(f'{name}: output={out.is_dir()} checkpoints={len(checkpoints)}')
    if not checkpoints:
        continue
    latest = checkpoints[-1]
    state_path = latest / 'trainer_state.json'
    print(f'  latest={latest.name} trainer_state={state_path.is_file()}')
    if not state_path.is_file():
        continue
    state = json.loads(state_path.read_text())
    losses = [entry.get('loss') for entry in state.get('log_history', []) if 'loss' in entry]
    finite = all(isinstance(value, (int, float)) and math.isfinite(value) for value in losses)
    print(f"  global_step={state.get('global_step')} loss_records={len(losses)} finite_losses={finite}")
    if losses:
        print(f'  recent_losses={losses[-5:]}')
PY

echo '=== fatal-signature audit of every workflow log ==='
mapfile -t TASK_LOGS < <(find "$TASK_LOG_DIR" -maxdepth 1 -type f -name '*.log' -print | sort)
echo "log_count=${#TASK_LOGS[@]}"
if [ "${#TASK_LOGS[@]}" -gt 0 ] && grep -H -E -i \
        'Traceback|CUDA out of memory|OutOfMemoryError|ChildFailedError|ProcessExitedException|NCCL[^[:cntrl:]]*(unhandled|system error|remote process exited|watchdog|collective operation timeout)|Segmentation fault|Bus error|nan loss|inf loss' \
        "${TASK_LOGS[@]}"; then
    echo 'FATAL_SIGNATURES_FOUND=YES'
else
    echo 'FATAL_SIGNATURES_FOUND=NO'
fi

echo '=== current experiment log progress tail ==='
for TASK_NAME in "${TASK_NAMES[@]}"; do
    TASK_LOG="$TASK_LOG_DIR/$TASK_NAME.log"
    echo "--- $TASK_NAME ---"
    if [ -s "$TASK_LOG" ]; then
        tail -c 400000 "$TASK_LOG" | tr '\r' '\n' |
            grep -E 'Embedding:|Total parameters:|Trainable parameters:|Running tokenizer|[0-9]+/[0-9]+|loss|Training completed|STOPPED|FAILED' |
            tail -30 || true
    else
        echo 'log=not_started'
    fi
done

echo '=== GPU state and ownership ==='
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory \
    --format=csv,noheader
echo 'TH2 READ-ONLY THREE-EXPERIMENT AUDIT COMPLETE'
