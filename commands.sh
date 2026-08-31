#1 +30+a
#th2-readonly-check-frequency-binned-ppl-a03-progress-20260831-a02
set -euo pipefail

TASK_ROOT=/mnt/local/_outputs/@PROJECT@/frequency_binned_ppl_four_10k_20260831_a03

echo '=== GPU snapshot ==='
date -u
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader

echo '=== by-token processes (read only) ==='
ps -eo pid=,ppid=,stat=,etime=,cmd= | grep '[p]pl_bytoken.py' || true

echo '=== shard artifacts and log tails ==='
for TASK_LOG in "$TASK_ROOT"/shards/*/*/run.log; do
  test -f "$TASK_LOG" || continue
  echo "--- $TASK_LOG"
  ls -lh "$(dirname "$TASK_LOG")"
  tail -8 "$TASK_LOG"
done

echo 'TH2 READONLY FREQUENCY-BINNED PPL A03 PROGRESS CHECK COMPLETE'
