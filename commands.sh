#1 +180+a
#th2-readonly-check-five-checkpoint-eval-and-partial-burn-20260904-a01
set -euo pipefail

TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_EVAL_LOG="$TASK_OUTPUT_BASE/eval_parallel_ranklift_hashedv2_btmos_steps_6500_10000_20260904_a01.log"
TASK_PARTIAL_BURN_LOG=/tmp/llm_pretrain_burn_eval_gpus_5_7.log

echo '=== live GPU utilization and process mapping ==='
date -u
hostname
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,power.draw \
    --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits

echo '=== expected live launchers ==='
pgrep -af '[e]val/eval_parallel.py.*--num-gpus 5'
pgrep -af '[e]val/eval_checkpoint.py.*--device cuda'
pgrep -af '[/]tmp/llm_pretrain_burn.py'

echo '=== require one compute process on every physical GPU ==='
TASK_ALL_PIDS=()
for TASK_GPU_INDEX in 0 1 2 3 4 5 6 7; do
    mapfile -t TASK_ONE_PIDS < <(
        nvidia-smi -i "$TASK_GPU_INDEX" --query-compute-apps=pid \
            --format=csv,noheader,nounits \
            | sed 's/^[[:space:]]*//;s/[[:space:]]*$//;/^$/d' | sort -nu
    )
    test "${#TASK_ONE_PIDS[@]}" -eq 1
    test "${TASK_ONE_PIDS[0]}" -ne 1
    TASK_ALL_PIDS+=("${TASK_ONE_PIDS[0]}")
    echo "gpu=$TASK_GPU_INDEX pid=${TASK_ONE_PIDS[0]}"
done
mapfile -t TASK_UNIQUE_PIDS < <(printf '%s\n' "${TASK_ALL_PIDS[@]}" | sort -nu)
test "${#TASK_UNIQUE_PIDS[@]}" -eq 8

echo '=== evaluation launcher and partial-burn communication progress ==='
test -s "$TASK_EVAL_LOG"
tail -80 "$TASK_EVAL_LOG"
test -s "$TASK_PARTIAL_BURN_LOG"
test "$(grep -Fc 'gpu_burn_ready' "$TASK_PARTIAL_BURN_LOG")" -eq 3
test "$(grep -Fc 'world_size=3' "$TASK_PARTIAL_BURN_LOG")" -eq 3
grep -Fq 'gpu_burn_progress' "$TASK_PARTIAL_BURN_LOG"
tail -30 "$TASK_PARTIAL_BURN_LOG"

echo '=== per-checkpoint eval log tails ==='
for TASK_CHECKPOINT in \
    "$TASK_OUTPUT_BASE/ranklift_tied_c124_m460/checkpoint-6500" \
    "$TASK_OUTPUT_BASE/ranklift_tied_c124_m460/checkpoint-10000" \
    "$TASK_OUTPUT_BASE/product_code_quota_h6144/checkpoint-6500" \
    "$TASK_OUTPUT_BASE/product_code_quota_h6144/checkpoint-10000" \
    "$TASK_OUTPUT_BASE/btmos_k3_c256_lb/checkpoint-6500"; do
    test -s "$TASK_CHECKPOINT/eval.log"
    echo "checkpoint=$TASK_CHECKPOINT"
    tail -12 "$TASK_CHECKPOINT/eval.log"
done
echo 'TH2 FIVE EVAL JOBS AND THREE-GPU COMMUNICATING PARTIAL BURN LIVE'
