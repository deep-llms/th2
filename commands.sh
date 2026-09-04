#1 +120+a
#th2-readonly-check-tiered-c512-training-start-20260904-a01
set -euo pipefail

TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_TIERED="$TASK_OUTPUT_BASE/tiered_ranklift_lb_t4_c512"
TASK_CONTROL="$TASK_OUTPUT_BASE/groupreduce_matched_lb_t4"
TASK_LOG_DIR="$TASK_OUTPUT_BASE/logs/tiered_c512_lb_groupreduce_10k_20260904_a01"

echo '=== read-only identity and GPU state ==='
date -u
hostname
nvidia-smi \
  --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader
nvidia-smi \
  --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader,nounits

echo '=== read-only process state ==='
pgrep -af '[r]un_experiments.py|[a]ccelerate.commands.launch|[t]rain_compositional.py' || true

echo '=== read-only output and progress state ==='
for TASK_OUTPUT in "$TASK_TIERED" "$TASK_CONTROL"; do
  echo "output=$TASK_OUTPUT"
  if [[ -d "$TASK_OUTPUT" ]]; then
    find "$TASK_OUTPUT" -mindepth 1 -maxdepth 1 -type d \
      -name 'checkpoint-*' -printf '%f\n' | sort -V | tail -5
    if [[ -s "$TASK_OUTPUT/train_progress.csv" ]]; then
      tail -5 "$TASK_OUTPUT/train_progress.csv"
    fi
  else
    echo 'not_created'
  fi
done

echo '=== read-only runner and active experiment logs ==='
test -s "$TASK_LOG_DIR/experiments.log"
tail -30 "$TASK_LOG_DIR/experiments.log"
if [[ -s "$TASK_LOG_DIR/tiered_ranklift_lb_t4_c512.log" ]]; then
  tail -50 "$TASK_LOG_DIR/tiered_ranklift_lb_t4_c512.log"
fi

echo 'TH2 TIERED-C512 TRAINING START READ-ONLY CHECK COMPLETE'
