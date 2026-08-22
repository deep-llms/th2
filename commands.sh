#1 +60+a
#th2-run-project-gpu-burn-20260822
set -euo pipefail

echo '=== th2 project GPU burn ==='
date -u
hostname

TASK_BURN_SCRIPT="scripts/gpu_burn.py"
test -s "$TASK_BURN_SCRIPT"

TASK_GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
echo "gpu_count=$TASK_GPU_COUNT"
test "$TASK_GPU_COUNT" -eq 8

echo '=== preflight: GPUs must be free ==='
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
TASK_EXISTING_GPU_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')"
test -z "$TASK_EXISTING_GPU_PIDS"

TASK_CHILD_PIDS=()
cleanup_burn_workers() {
    trap - EXIT INT TERM
    if [ "${#TASK_CHILD_PIDS[@]}" -gt 0 ]; then
        kill -TERM "${TASK_CHILD_PIDS[@]}" 2>/dev/null || true
        wait "${TASK_CHILD_PIDS[@]}" 2>/dev/null || true
    fi
}
trap cleanup_burn_workers EXIT INT TERM

echo '=== launch one supervised worker per GPU ==='
for TASK_GPU in 0 1 2 3 4 5 6 7; do
    TASK_LOG="/tmp/project_gpu_burn_gpu${TASK_GPU}.log"
    env CUDA_VISIBLE_DEVICES="$TASK_GPU" /usr/bin/python3 -u "$TASK_BURN_SCRIPT" >"$TASK_LOG" 2>&1 &
    TASK_CHILD_PIDS+=("$!")
    echo "launched gpu=$TASK_GPU pid=$! log=$TASK_LOG"
done

sleep 30

echo '=== post-launch verification ==='
for TASK_PID in "${TASK_CHILD_PIDS[@]}"; do
    kill -0 "$TASK_PID"
done
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
TASK_GPU_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d' | sort -nu)"
TASK_GPU_PROCESS_COUNT="$(printf '%s\n' "$TASK_GPU_PIDS" | sed '/^[[:space:]]*$/d' | wc -l)"
echo "gpu_compute_process_count=$TASK_GPU_PROCESS_COUNT"
test "$TASK_GPU_PROCESS_COUNT" -eq 8

for TASK_PID in $TASK_GPU_PIDS; do
    TASK_CMDLINE="$(tr '\0' ' ' < "/proc/$TASK_PID/cmdline")"
    printf '%s\n' "$TASK_CMDLINE" | grep -Fq "$TASK_BURN_SCRIPT"
done

echo 'TH2 GPU BURN VERIFIED: RUNNING ON ALL 8 GPUS'
echo '=== supervising workers; this command intentionally remains running ==='
set +e
wait -n "${TASK_CHILD_PIDS[@]}"
TASK_FIRST_EXIT=$?
set -e
echo "ERROR: a burn worker exited with status $TASK_FIRST_EXIT"
exit 1
