#1 +90+a
#th2-readonly-final-rse-hashed-completion-and-burn-20260901-a01
set -euo pipefail

TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_LOG_DIR="$TASK_OUTPUT_BASE/logs/final_rse_b8a8_hashed_10k_20260831_a02"
TASK_PYTHON=/mnt/local/conda-py311/envs/sparse_emb/bin/python3.11

echo '=== timestamp and GPU state ==='
date -u
hostname
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader

echo '=== GPU compute processes ==='
mapfile -t TASK_GPU_PIDS < <(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    | sed '/^[[:space:]]*$/d;s/[[:space:]]//g' | sort -nu
)
echo "unique_gpu_pid_count=${#TASK_GPU_PIDS[@]}"
for TASK_PID in "${TASK_GPU_PIDS[@]}"; do
  if [ -r "/proc/$TASK_PID/status" ] && [ -r "/proc/$TASK_PID/cmdline" ]; then
    TASK_PPID="$(awk '/^PPid:/ {print $2}' "/proc/$TASK_PID/status" || true)"
    printf 'pid=%s ppid=%s cmd=' "$TASK_PID" "$TASK_PPID"
    tr '\0' ' ' < "/proc/$TASK_PID/cmdline" || true
    echo
  else
    echo "pid=$TASK_PID exited_during_read=true"
  fi
done

echo '=== relevant tmux sessions ==='
tmux list-sessions 2>/dev/null | grep -E 'final-rse|hashed|experiment|burn' || true

echo '=== experiment checkpoint states ==='
for TASK_NAME in \
  residual_subspace_experts_tied_g12_r120_q80 \
  product_code_hashed_h2048; do
  TASK_OUTPUT="$TASK_OUTPUT_BASE/$TASK_NAME"
  echo "experiment=$TASK_NAME"
  if [ ! -d "$TASK_OUTPUT" ]; then
    echo 'output=ABSENT'
    continue
  fi
  TASK_LATEST="$({ find "$TASK_OUTPUT" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' || true; } | sort -V | tail -1)"
  echo "latest_checkpoint=${TASK_LATEST:-NONE}"
  if [ -n "$TASK_LATEST" ] && [ -s "$TASK_OUTPUT/$TASK_LATEST/trainer_state.json" ]; then
    "$TASK_PYTHON" - "$TASK_OUTPUT/$TASK_LATEST/trainer_state.json" <<'PY'
import json
import math
import sys

with open(sys.argv[1]) as handle:
    state = json.load(handle)
rows = [row for row in state.get("log_history", []) if "loss" in row]
print(f"global_step={state.get('global_step')} epoch={state.get('epoch')}")
print(f"finite_loss_rows={sum(math.isfinite(float(row['loss'])) for row in rows)}/{len(rows)}")
print(f"last_loss_row={rows[-1] if rows else None}")
PY
  fi
done

echo '=== experiment runner summary ==='
if [ -s "$TASK_LOG_DIR/experiments.log" ]; then
  tail -80 "$TASK_LOG_DIR/experiments.log"
else
  echo 'experiments.log=ABSENT'
fi

echo '=== fatal-signature audit ==='
TASK_FATAL=0
for TASK_LOG in \
  "$TASK_LOG_DIR/residual_subspace_experts_tied_g12_r120_q80.log" \
  "$TASK_LOG_DIR/product_code_hashed_h2048.log"; do
  echo "log=$TASK_LOG"
  if [ ! -s "$TASK_LOG" ]; then
    echo 'state=ABSENT'
    continue
  fi
  if grep -E -i 'Traceback|CUDA out of memory|OutOfMemoryError|RuntimeError:|NCCL[^[:cntrl:]]*(unhandled|system error|remote process exited|watchdog|collective operation timeout)' "$TASK_LOG"; then
    TASK_FATAL=1
  else
    echo 'fatal_signatures=NONE'
  fi
  echo 'latest_log_lines:'
  tail -c 200000 "$TASK_LOG" | tr '\r' '\n' | tail -20
done
echo "fatal_signature_flag=$TASK_FATAL"

echo '=== burn handoff state ==='
if [ -s /tmp/llm_pretrain_burn.py ]; then
  sha256sum /tmp/llm_pretrain_burn.py
else
  echo 'burn_script=ABSENT'
fi
if [ -s /tmp/llm_pretrain_burn_launcher.pid ]; then
  echo "burn_launcher_pid=$(cat /tmp/llm_pretrain_burn_launcher.pid)"
fi
if [ -s /tmp/llm_pretrain_burn_after_final_rse_b8a8_hashed_10k_a02.log ]; then
  grep -E 'gpu_burn_ready|collective_probe_sum|gpu_burn_progress' \
    /tmp/llm_pretrain_burn_after_final_rse_b8a8_hashed_10k_a02.log | tail -24
else
  echo 'final_burn_log=ABSENT'
fi

echo 'TH2 READONLY FINAL RSE HASHED STATUS CHECK COMPLETE'
