#1 +60+a
#th2-check-pure-local-g16-r128-runtime-20260825
set -euo pipefail

TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_OUTPUT_DIR="$TASK_OUTPUT_BASE/pure_local_tied_g16_r128"
TASK_LOG_DIR="$TASK_OUTPUT_BASE/logs/pure_local_tied_g16_r128_20260825"

echo '=== Pure-local G16 R128 runtime check ==='
date -u
hostname

echo '=== GPU state ==='
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
mapfile -t TASK_GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -u
)
test "${#TASK_GPU_PIDS[@]}" -eq 8 || {
    echo "ERROR: expected 8 training GPU processes, found ${#TASK_GPU_PIDS[@]}"
    exit 1
}

for TASK_PID in "${TASK_GPU_PIDS[@]}"; do
    test -r "/proc/$TASK_PID/cmdline"
    TASK_CMDLINE="$(tr '\0' ' ' < "/proc/$TASK_PID/cmdline")"
    echo "gpu_pid=$TASK_PID cmd=$TASK_CMDLINE"
    case "$TASK_CMDLINE" in
        *train_compositional.py*"--arm pure_local"*"--pure_local_rank 128"*"--num_groups 16"*"--tie_output"*) ;;
        *) echo "ERROR: unexpected GPU process $TASK_PID"; exit 1 ;;
    esac
done
echo 'TH2 PURE-LOCAL HAS 8 EXPECTED GPU WORKERS'

echo '=== output and log paths ==='
test -d "$TASK_OUTPUT_DIR"
test -d "$TASK_LOG_DIR"
du -sh "$TASK_OUTPUT_DIR" "$TASK_LOG_DIR"
find "$TASK_OUTPUT_DIR" -maxdepth 2 -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' \
    | sort | tail -n 30
find "$TASK_LOG_DIR" -maxdepth 2 -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' \
    | sort | tail -n 30

echo '=== recent logs ==='
while IFS= read -r TASK_LOG_FILE; do
    echo "--- $TASK_LOG_FILE"
    tail -n 80 "$TASK_LOG_FILE"
done < <(find "$TASK_LOG_DIR" "$TASK_OUTPUT_DIR" -maxdepth 2 -type f -name '*.log' | sort)

echo 'TH2 PURE-LOCAL RUNTIME CHECK OK'
