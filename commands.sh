#1 +120+a
#th2-replace-pure-local-handoff-watcher-after-audit-20260825
set -euo pipefail

echo '=== replace only the Pure-local handoff watcher with audited version ==='
date -u
hostname

TASK_PROJECT_DIR="$PWD"
TASK_WATCHER="$TASK_PROJECT_DIR/scripts/watch_pure_local_10k_eval_finetune_burn.sh"
TASK_TRAIN_OUTPUT=/mnt/local/_outputs/@PROJECT@/pure_local_tied_g16_r128
TASK_EVAL_PYTHON=/mnt/local/conda-py311/envs/eval/bin/python3.11
test -s "$TASK_WATCHER"
test -x "$TASK_EVAL_PYTHON"

find_exact_watchers() {
    local proc
    local -a argv
    for proc in /proc/[0-9]*; do
        [[ -r "$proc/cmdline" ]] || continue
        argv=()
        mapfile -d '' -t argv < "$proc/cmdline" || true
        [[ "${#argv[@]}" -ge 2 ]] || continue
        if [[ "${argv[0]}" == bash || "${argv[0]}" == */bash ]] \
                && [[ "${argv[1]}" == "$TASK_WATCHER" ]]; then
            printf '%s\n' "${proc#/proc/}"
        fi
    done | sort -nu
}

echo '=== stop the old watcher only; do not touch training ==='
mapfile -t TASK_OLD_WATCHERS < <(find_exact_watchers)
[[ "${#TASK_OLD_WATCHERS[@]}" -le 1 ]] \
    || { echo "ERROR: multiple old watchers found: ${TASK_OLD_WATCHERS[*]}"; exit 1; }
if [[ "${#TASK_OLD_WATCHERS[@]}" -eq 1 ]]; then
    echo "stopping_old_watcher_pid=${TASK_OLD_WATCHERS[0]}"
    kill -TERM "${TASK_OLD_WATCHERS[0]}"
    for TASK_WAIT in $(seq 1 30); do
        mapfile -t TASK_REMAINING_WATCHERS < <(find_exact_watchers)
        [[ "${#TASK_REMAINING_WATCHERS[@]}" -eq 0 ]] && break
        sleep 1
    done
fi
mapfile -t TASK_REMAINING_WATCHERS < <(find_exact_watchers)
[[ "${#TASK_REMAINING_WATCHERS[@]}" -eq 0 ]] \
    || { echo "ERROR: old watcher remains: ${TASK_REMAINING_WATCHERS[*]}"; exit 1; }
echo 'OLD WATCHER STOPPED; TRAINING WAS NOT SIGNALED'

echo '=== prove the eight exact Pure-local GPU workers are still running ==='
mapfile -t TASK_GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -nu
)
[[ "${#TASK_GPU_PIDS[@]}" -eq 8 ]] \
    || { echo "ERROR: expected 8 training workers, found ${#TASK_GPU_PIDS[@]}"; exit 1; }
for TASK_PID in "${TASK_GPU_PIDS[@]}"; do
    test -r "/proc/$TASK_PID/cmdline"
    TASK_CMDLINE=$(tr '\0' ' ' < "/proc/$TASK_PID/cmdline")
    echo "training_gpu_pid=$TASK_PID cmd=$TASK_CMDLINE"
    [[ "$TASK_CMDLINE" == *train_compositional.py* \
        && "$TASK_CMDLINE" == *"--output_dir $TASK_TRAIN_OUTPUT"* \
        && "$TASK_CMDLINE" == *"--arm pure_local"* \
        && "$TASK_CMDLINE" == *"--pure_local_rank 128"* \
        && "$TASK_CMDLINE" == *"--num_groups 16"* \
        && "$TASK_CMDLINE" == *"--tie_output"* ]] \
        || { echo "ERROR: unexpected GPU process $TASK_PID"; exit 1; }
done
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
echo 'PURE-LOCAL TRAINING UNINTERRUPTED WITH ALL 8 EXPECTED WORKERS'

export SPARSE_EMB_PROJECT_DIR="$TASK_PROJECT_DIR"
export SPARSE_EMB_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
export SPARSE_EMB_MODEL_DIR=/mnt/local/_models/@PROJECT@/Qwen3-0.6B
export SPARSE_EMB_EVAL_DIR=/mnt/local/_data/@PROJECT@/data/Qwen_Qwen3-0.6B/eval
export SPARSE_EMB_BENCH_ROOT=/mnt/local/_data/@PROJECT@/benchmarks/hf
export SPARSE_EMB_EVAL_PYTHON="$TASK_EVAL_PYTHON"

echo '=== start audited fail-closed watcher ==='
exec bash "$TASK_WATCHER"
