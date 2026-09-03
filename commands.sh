#1 +60+a
#th2-readonly-verify-three-experiment-training-start-20260903-a02
set -euo pipefail

TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_LOG_DIR="$TASK_OUTPUT_BASE/logs/ranklift_hashedv2_btmos_10k_20260903_a01"
TASK_EXPERIMENT_LOG="$TASK_LOG_DIR/experiments.log"
TASK_RANKLIFT_LOG="$TASK_LOG_DIR/ranklift_tied_c124_m460.log"

echo '=== identity ==='
date -u
hostname

echo '=== exact sequential runner ==='
mapfile -t TASK_RUNNER_PIDS < <(pgrep -f '[r]un_experiments.py' | sort -nu)
echo "runner_count=${#TASK_RUNNER_PIDS[@]}"
for TASK_PID in "${TASK_RUNNER_PIDS[@]}"; do
    ps -p "$TASK_PID" -o pid=,ppid=,etimes=,stat=,args=
done

echo '=== Accelerate and training processes ==='
pgrep -af '[a]ccelerate.commands.launch|[a]ccelerate launch' \
    || echo 'accelerate_launcher=absent'
mapfile -t TASK_TRAIN_PIDS < <(pgrep -f '[t]rain_compositional.py' | sort -nu)
echo "train_process_count=${#TASK_TRAIN_PIDS[@]}"
for TASK_PID in "${TASK_TRAIN_PIDS[@]}"; do
    ps -p "$TASK_PID" -o pid=,ppid=,etimes=,stat=,args=
done

echo '=== current RankLift output ==='
if [ -d "$TASK_OUTPUT_BASE/ranklift_tied_c124_m460" ]; then
    du -sh "$TASK_OUTPUT_BASE/ranklift_tied_c124_m460"
    find "$TASK_OUTPUT_BASE/ranklift_tied_c124_m460" \
        -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' \
        -printf '%f\n' | sort -V | tail -10
else
    echo 'ranklift_output=not_created_yet'
fi

echo '=== sequential runner log ==='
test -s "$TASK_EXPERIMENT_LOG"
tail -50 "$TASK_EXPERIMENT_LOG"

echo '=== RankLift training log health and progress ==='
test -s "$TASK_RANKLIFT_LOG"
if grep -E -i 'Traceback|CUDA out of memory|OutOfMemoryError|ChildFailedError|ProcessExitedException|NCCL[^[:cntrl:]]*(unhandled|system error|remote process exited|watchdog|collective operation timeout)|Segmentation fault|Bus error' \
        "$TASK_RANKLIFT_LOG"; then
    echo 'ERROR: fatal signature found in RankLift log' >&2
    exit 1
fi
tail -c 300000 "$TASK_RANKLIFT_LOG" | tr '\r' '\n' |
    grep -E 'Embedding:|Total parameters:|Trainable parameters:|[0-9]+/[0-9]+|loss' |
    tail -40 || true

echo '=== GPU state ==='
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory \
    --format=csv,noheader

echo 'TH2 THREE-EXPERIMENT TRAINING START SNAPSHOT OK'
