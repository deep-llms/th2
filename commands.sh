#1 +45+a
#th2-readonly-ranklift-10k-live-audit-20260902-a01
set -euo pipefail

TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_LOG="$TASK_OUTPUT_BASE/logs/ranklift_tied_c124_m460_10k_20260902/ranklift_tied_c124_m460.log"

echo '=== gpu utilization ==='
date -u
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits

echo '=== training process commands (read only) ==='
mapfile -t TASK_PIDS < <(pgrep -f 'train_compositional.py' || true)
for TASK_PID in "${TASK_PIDS[@]}"; do
    [[ -r "/proc/$TASK_PID/cmdline" ]] || continue
    printf 'pid=%s ppid=%s cmd=' "$TASK_PID" "$(awk '/^PPid:/{print $2}' "/proc/$TASK_PID/status")"
    tr '\0' ' ' < "/proc/$TASK_PID/cmdline"
    printf '\n'
done

echo '=== live RankLift log tail ==='
test -s "$TASK_LOG"
tr '\r' '\n' < "$TASK_LOG" | tail -n 120

echo '=== narrow fatal scan ==='
if grep -niE 'Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|ChildFailedError|ProcessExitedException|Segmentation fault|Bus error' "$TASK_LOG"; then
    echo 'FATAL_SIGNATURE_FOUND' >&2
    exit 1
fi
echo 'NO_FATAL_SIGNATURE_FOUND'
echo 'TH2 RANKLIFT 10K LIVE AUDIT COMPLETE; TRAINING UNMODIFIED'
