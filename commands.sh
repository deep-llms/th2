#1 +30+a
#th2-readonly-gpu-guard-logs-20260911-a01
set -euo pipefail
date -u
hostname
shopt -s nullglob
TASK_LOGS=(/mnt/local/_gpu_guard/burn-*.log)
printf 'MATCHING_GPU_GUARD_LOGS=%s\n' "${#TASK_LOGS[@]}"
for TASK_LOG in "${TASK_LOGS[@]}"; do
    printf '\nGPU_GUARD_LOG: %s\n' "$TASK_LOG"
    stat --format='size=%s bytes modified=%y' -- "$TASK_LOG"
    sha256sum -- "$TASK_LOG"
    tail -n 200 -- "$TASK_LOG"
done
date -u
echo GPU_GUARD_READONLY_LOG_REVIEW_COMPLETE
