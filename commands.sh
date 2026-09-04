#1 +240+a
#th2-stop-btmos-safely-and-restore-burn-20260904-a01
set -euo pipefail

die() {
    echo "ERROR: $*" >&2
    exit 1
}

read_gpu_pids() {
    local output
    output="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" \
        || return 1
    printf '%s\n' "$output" |
        sed 's/^[[:space:]]*//;s/[[:space:]]*$//;/^$/d' | sort -nu
}

TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_BTMOS_OUTPUT="$TASK_OUTPUT_BASE/btmos_k3_c256_lb"
TASK_LOG_DIR="$TASK_OUTPUT_BASE/logs/ranklift_hashedv2_btmos_10k_20260903_a01"
TASK_BURN_LOG="$TASK_OUTPUT_BASE/logs/gpu_burn_after_cancelled_btmos_20260904_a01.log"

echo '=== pre-stop state ==='
date -u
test -s /tmp/llm_pretrain_burn.py \
    || die 'runner-provided burn script is missing or empty'
test -d "$TASK_BTMOS_OUTPUT" || die 'BT-MoS output directory is absent'
mapfile -t TASK_GPU_NAMES < <(
    nvidia-smi --query-gpu=name --format=csv,noheader |
        sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
)
[[ "${#TASK_GPU_NAMES[@]}" -eq 8 ]] \
    || die "expected 8 GPUs, found ${#TASK_GPU_NAMES[@]}"
for TASK_GPU_INDEX in "${!TASK_GPU_NAMES[@]}"; do
    [[ "${TASK_GPU_NAMES[$TASK_GPU_INDEX]}" == *B200* ]] \
        || die "GPU $TASK_GPU_INDEX is not B200: ${TASK_GPU_NAMES[$TASK_GPU_INDEX]}"
done
pgrep -af '[r]un_experiments.py.*--stop-at-step 10000' \
    || die 'expected sequential runner is absent'
pgrep -af '[a]ccelerate.commands.launch.*btmos_k3_c256_lb' \
    || die 'expected BT-MoS Accelerate launcher is absent'

mapfile -t TASK_GPU_PIDS < <(read_gpu_pids)
[[ "${#TASK_GPU_PIDS[@]}" -eq 8 ]] \
    || die "expected exactly 8 BT-MoS GPU ranks, found ${#TASK_GPU_PIDS[@]}"

TASK_PER_GPU_PIDS=()
for TASK_GPU_INDEX in 0 1 2 3 4 5 6 7; do
    mapfile -t TASK_ONE_GPU_PIDS < <(
        nvidia-smi -i "$TASK_GPU_INDEX" \
            --query-compute-apps=pid --format=csv,noheader,nounits |
            sed 's/^[[:space:]]*//;s/[[:space:]]*$//;/^$/d' | sort -nu
    )
    [[ "${#TASK_ONE_GPU_PIDS[@]}" -eq 1 ]] \
        || die "GPU $TASK_GPU_INDEX does not have exactly one training rank"
    TASK_PER_GPU_PIDS+=("${TASK_ONE_GPU_PIDS[0]}")
done
mapfile -t TASK_UNIQUE_PER_GPU_PIDS < <(
    printf '%s\n' "${TASK_PER_GPU_PIDS[@]}" | sort -nu
)
[[ "${#TASK_UNIQUE_PER_GPU_PIDS[@]}" -eq 8 ]] \
    || die 'the eight GPUs do not have eight distinct rank PIDs'
[[ "${TASK_UNIQUE_PER_GPU_PIDS[*]}" == "${TASK_GPU_PIDS[*]}" ]] \
    || die 'per-GPU PID set differs from the all-GPU PID set'

echo '=== validate every target before signalling ==='
for TASK_PID in "${TASK_GPU_PIDS[@]}"; do
    [[ "$TASK_PID" =~ ^[0-9]+$ ]] || die "invalid PID: $TASK_PID"
    [[ "$TASK_PID" -ne 1 ]] || die 'refusing to signal PID 1'
    [[ -r "/proc/$TASK_PID/cmdline" ]] || die "cannot inspect PID $TASK_PID"
    TASK_CMDLINE="$(tr '\0' ' ' < "/proc/$TASK_PID/cmdline")"
    [[ "$TASK_CMDLINE" == *train_compositional.py* ]] \
        || die "PID $TASK_PID is not train_compositional.py: $TASK_CMDLINE"
    [[ "$TASK_CMDLINE" == *"--output_dir $TASK_BTMOS_OUTPUT"* ]] \
        || die "PID $TASK_PID has wrong output directory: $TASK_CMDLINE"
    [[ "$TASK_CMDLINE" == *'--mos_components 3'* ]] \
        || die "PID $TASK_PID is not the expected BT-MoS run: $TASK_CMDLINE"
    TASK_PPID="$(awk '/^PPid:/ {print $2}' "/proc/$TASK_PID/status")"
    [[ "$TASK_PPID" =~ ^[0-9]+$ && "$TASK_PPID" -ne 1 ]] \
        || die "unsafe parent PID for rank $TASK_PID: $TASK_PPID"
    TASK_PARENT_CMDLINE="$(tr '\0' ' ' < "/proc/$TASK_PPID/cmdline")"
    [[ "$TASK_PARENT_CMDLINE" == *accelerate.commands.launch* ]] \
        || die "unexpected parent for rank $TASK_PID: $TASK_PARENT_CMDLINE"
    echo "verified_btmos_rank gpu_pid=$TASK_PID parent=$TASK_PPID"
done
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader

echo '=== stop only the eight verified BT-MoS GPU ranks ==='
kill -9 "${TASK_GPU_PIDS[@]}" 2>/dev/null || true

echo '=== wait for BT-MoS launcher and sequential runner to exit ==='
for TASK_ATTEMPT in $(seq 1 36); do
    if ! pgrep -f '[t]rain_compositional.py' >/dev/null \
            && ! pgrep -f '[a]ccelerate.commands.launch' >/dev/null \
            && ! pgrep -f '[r]un_experiments.py.*--stop-at-step 10000' >/dev/null; then
        echo "training_processes_gone_attempt=$TASK_ATTEMPT"
        break
    fi
    [[ "$TASK_ATTEMPT" -lt 36 ]] \
        || die 'training processes still exist after 180 seconds'
    sleep 5
done

echo '=== wait for the original workflow exit trap to restore burns ==='
for TASK_ATTEMPT in $(seq 1 24); do
    mapfile -t TASK_AFTER_PIDS < <(read_gpu_pids)
    if [[ "${#TASK_AFTER_PIDS[@]}" -eq 8 ]]; then
        echo "eight_gpu_processes_restored_attempt=$TASK_ATTEMPT"
        break
    fi
    if [[ "${#TASK_AFTER_PIDS[@]}" -ne 0 ]]; then
        die "unexpected partial GPU ownership after stop: ${TASK_AFTER_PIDS[*]}"
    fi
    if [[ "$TASK_ATTEMPT" -eq 24 ]]; then
        echo 'original workflow did not restore burns; starting runner-provided burn'
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
            python3 /tmp/llm_pretrain_burn.py >>"$TASK_BURN_LOG" 2>&1 &
        sleep 30
        mapfile -t TASK_AFTER_PIDS < <(read_gpu_pids)
        [[ "${#TASK_AFTER_PIDS[@]}" -eq 8 ]] \
            || die "expected 8 burn workers after fallback start, found ${#TASK_AFTER_PIDS[@]}"
        break
    fi
    sleep 5
done

echo '=== validate exactly one runner burn worker per GPU ==='
for TASK_GPU_INDEX in 0 1 2 3 4 5 6 7; do
    mapfile -t TASK_ONE_GPU_PIDS < <(
        nvidia-smi -i "$TASK_GPU_INDEX" \
            --query-compute-apps=pid --format=csv,noheader,nounits |
            sed 's/^[[:space:]]*//;s/[[:space:]]*$//;/^$/d' | sort -nu
    )
    [[ "${#TASK_ONE_GPU_PIDS[@]}" -eq 1 ]] \
        || die "GPU $TASK_GPU_INDEX does not have exactly one burn worker"
    TASK_PID="${TASK_ONE_GPU_PIDS[0]}"
    [[ "$TASK_PID" -ne 1 ]] || die 'burn worker unexpectedly has PID 1'
    TASK_PPID="$(awk '/^PPid:/ {print $2}' "/proc/$TASK_PID/status")"
    TASK_PARENT_CMDLINE="$(tr '\0' ' ' < "/proc/$TASK_PPID/cmdline")"
    [[ "$TASK_PARENT_CMDLINE" == *'/tmp/llm_pretrain_burn.py'* ]] \
        || die "GPU $TASK_GPU_INDEX process is not owned by runner burn: $TASK_PARENT_CMDLINE"
    echo "gpu=$TASK_GPU_INDEX burn_worker_pid=$TASK_PID burn_parent_pid=$TASK_PPID"
done

echo '=== preserved BT-MoS checkpoint and final state ==='
find "$TASK_BTMOS_OUTPUT" -mindepth 1 -maxdepth 1 -type d \
    -name 'checkpoint-*' -printf '%f\n' | sort -V | tail -5
tail -25 "$TASK_LOG_DIR/experiments.log"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader
echo 'TH2 BTMOS STOPPED SAFELY; CHECKPOINTS PRESERVED; ALL 8 BURNS ACTIVE'
