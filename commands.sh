#1 +120+a
#th2-stop-clean-matched-global-lr-dense-tied-b200-20260824
set -euo pipefail

TASK_PROJECT_DIR="$PWD"
TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_DATA_DIR=/mnt/local/_data/@PROJECT@/data/Qwen_Qwen3-0.6B/train
TASK_GLOBAL_OUTPUT="$TASK_OUTPUT_BASE/global_lowrank_tied_r128_b200"
TASK_DENSE_OUTPUT="$TASK_OUTPUT_BASE/dense_tied_baseline_b200"
TASK_LOG_DIR="$TASK_OUTPUT_BASE/logs/b200_matched_global_lr_then_dense_tied_20260824"
TASK_HF_DATASETS_CACHE=/mnt/local/.cache/huggingface/datasets

is_current_training_process() {
    local cmd=$1
    [[ "$cmd" == *"run_experiments.py --experiments 13 14"* ]] \
        || [[ "$cmd" == *"global_lowrank_tied_r128_b200"* ]] \
        || [[ "$cmd" == *"dense_tied_baseline_b200"* ]]
}

current_process_pids() {
    local proc pid cmd
    for proc in /proc/[0-9]*; do
        pid=${proc#/proc/}
        [ -r "$proc/cmdline" ] || continue
        cmd=$(tr '\0' ' ' < "$proc/cmdline" 2>/dev/null || true)
        if is_current_training_process "$cmd"; then
            printf '%s\n' "$pid"
        fi
    done | sort -nu
}

echo '=== identify the exact current matched-control processes ==='
date -u
hostname
mapfile -t TASK_PIDS < <(current_process_pids)
test "${#TASK_PIDS[@]}" -gt 0 || {
    echo 'ERROR: no current global-LR/dense matched-control processes found'
    exit 1
}
for pid in "${TASK_PIDS[@]}"; do
    printf 'pid=%s cmd=' "$pid"
    tr '\0' ' ' < "/proc/$pid/cmdline" || true
    echo
done

echo '=== stop runner, launcher, and all eight workers ==='
kill -TERM "${TASK_PIDS[@]}" 2>/dev/null || true
for attempt in 1 2 3; do
    sleep 5
    mapfile -t TASK_PIDS < <(current_process_pids)
    [ "${#TASK_PIDS[@]}" -eq 0 ] && break
    echo "remaining_after_term_attempt_${attempt}=${TASK_PIDS[*]}"
    kill -TERM "${TASK_PIDS[@]}" 2>/dev/null || true
done
mapfile -t TASK_PIDS < <(current_process_pids)
if [ "${#TASK_PIDS[@]}" -gt 0 ]; then
    echo "force_kill=${TASK_PIDS[*]}"
    kill -KILL "${TASK_PIDS[@]}" 2>/dev/null || true
    sleep 5
fi
mapfile -t TASK_PIDS < <(current_process_pids)
test "${#TASK_PIDS[@]}" -eq 0

echo '=== verify no unexpected GPU process remains before cleanup ==='
mapfile -t TASK_GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -u
)
if [ "${#TASK_GPU_PIDS[@]}" -ne 0 ]; then
    echo "ERROR: GPU processes remain after stopping the current run: ${TASK_GPU_PIDS[*]}"
    for pid in "${TASK_GPU_PIDS[@]}"; do
        printf 'pid=%s cmd=' "$pid"
        tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true
        echo
    done
    exit 1
fi

echo '=== inventory scoped artifacts before deletion ==='
for path in "$TASK_GLOBAL_OUTPUT" "$TASK_DENSE_OUTPUT" "$TASK_LOG_DIR"; do
    if [ -e "$path" ]; then
        du -sh "$path"
    else
        echo "absent=$path"
    fi
done
if [ -d "$TASK_HF_DATASETS_CACHE" ]; then
    du -sh "$TASK_HF_DATASETS_CACHE"
else
    echo "absent=$TASK_HF_DATASETS_CACHE"
fi
TASK_DATA_CACHE_COUNT=$(find "$TASK_DATA_DIR" -type f -name 'cache-*' | wc -l)
echo "dataset_preprocessing_cache_files=$TASK_DATA_CACHE_COUNT"

echo '=== remove only current outputs/logs and preprocessing caches ==='
rm -rf -- "$TASK_GLOBAL_OUTPUT"
rm -rf -- "$TASK_DENSE_OUTPUT"
rm -rf -- "$TASK_LOG_DIR"
rm -rf -- "$TASK_HF_DATASETS_CACHE"
find "$TASK_DATA_DIR" -type f -name 'cache-*' -delete

echo '=== remove only the current offline W&B run if present ==='
if [ -d "$TASK_PROJECT_DIR/wandb" ]; then
    mapfile -t TASK_WANDB_DIRS < <(
        find "$TASK_PROJECT_DIR/wandb" -mindepth 1 -maxdepth 1 -type d \
            -name 'offline-run-20260824_1942*' -print
    )
    for path in "${TASK_WANDB_DIRS[@]}"; do
        case "$(basename "$path")" in
            offline-run-20260824_1942*) rm -rf -- "$path" ;;
            *) echo "REFUSE unexpected W&B path: $path"; exit 1 ;;
        esac
    done
fi

echo '=== final verification ==='
for path in "$TASK_GLOBAL_OUTPUT" "$TASK_DENSE_OUTPUT" "$TASK_LOG_DIR" \
            "$TASK_HF_DATASETS_CACHE"; do
    test ! -e "$path"
    echo "removed=$path"
done
test "$(find "$TASK_DATA_DIR" -type f -name 'cache-*' | wc -l)" -eq 0
echo 'dataset_preprocessing_cache_files=0'
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
mapfile -t TASK_GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -u
)
test "${#TASK_GPU_PIDS[@]}" -eq 0
echo 'TH2 CURRENT MATCHED-CONTROL RUN STOPPED CLEANED ALL 8 B200 GPUS FREE'
