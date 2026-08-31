#1 +180+a
#th2-cancel-clean-final-rse-hashed-and-dataset-caches-20260831-a01
set -euo pipefail

TASK_PROJECT=/mnt/local/@PROJECT@
TASK_PYTHON=/mnt/local/conda-py311/envs/sparse_emb/bin/python3.11
TASK_DATA_ROOT=/mnt/local/_data/@PROJECT@
TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_RSE_OUTPUT="$TASK_OUTPUT_BASE/residual_subspace_experts_tied_g12_r120_q80"
TASK_HASHED_OUTPUT="$TASK_OUTPUT_BASE/product_code_hashed_h2048"
TASK_MAIN_LOG_DIR="$TASK_OUTPUT_BASE/logs/final_rse_hashed_10k_20260831_a01"
TASK_RETRY_LOG_DIR="$TASK_OUTPUT_BASE/logs/final_rse_retry_b8a8_10k_20260831_a01"
TASK_RSE_WANDB="$TASK_PROJECT/wandb/offline-run-20260831_203542-4tmxhqqr"
TASK_HASHED_WANDB="$TASK_PROJECT/wandb/offline-run-20260831_203724-vqibaqes"
TASK_MAIN_JOB=th2-train-final-rse-then-hashed-10k-20260831-a01
TASK_RETRY_JOB=th2-wait-hashed-then-retry-rse-b8a8-10k-and-burn-20260831-a01

gpu_pids() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    | sed '/^[[:space:]]*$/d;s/[[:space:]]//g' | sort -nu
}

echo '=== cancellation identity and current state ==='
date -u
hostname
cd "$TASK_PROJECT"
pwd
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader
tmux list-sessions -F '#{session_name}' 2>/dev/null || true
ps -eo pid,ppid,pgid,stat,args | grep -E \
  'run_experiments.py|train_compositional.py|train_product_code_tied|train_residual_subspace_experts|final_rse_hashed|final_rse_retry' \
  | grep -v grep || true

echo '=== stop the exact main and retry tmux sessions if present ==='
mapfile -t TASK_TMUX_SESSIONS < <(tmux list-sessions -F '#{session_name}' 2>/dev/null || true)
for TASK_SESSION in "${TASK_TMUX_SESSIONS[@]}"; do
  case "$TASK_SESSION" in
    *"$TASK_MAIN_JOB"*|*"$TASK_RETRY_JOB"*)
      echo "killing_tmux_session=$TASK_SESSION"
      tmux kill-session -t "$TASK_SESSION"
      ;;
  esac
done
sleep 5

echo '=== stop only GPU workers belonging to the requested Hashed/RSE runs ==='
for TASK_ATTEMPT in 1 2 3; do
  mapfile -t TASK_GPU_PIDS < <(gpu_pids)
  (( ${#TASK_GPU_PIDS[@]} == 0 )) && break
  TASK_ALL_OWNED=1
  for TASK_PID in "${TASK_GPU_PIDS[@]}"; do
    test "$TASK_PID" != 1
    TASK_CMD="$(tr '\0' ' ' <"/proc/$TASK_PID/cmdline" 2>/dev/null || true)"
    echo "gpu_pid=$TASK_PID cmd=$TASK_CMD"
    if [[ "$TASK_CMD" != *train_compositional.py* ]] || \
       { [[ "$TASK_CMD" != *'--arm product_code'* ]] && \
         [[ "$TASK_CMD" != *'--arm residual_subspace_experts'* ]]; }; then
      TASK_ALL_OWNED=0
    fi
  done
  if [[ "$TASK_ALL_OWNED" -ne 1 ]]; then
    echo 'ERROR: refusing to kill an unidentified GPU process' >&2
    exit 1
  fi
  "$TASK_PYTHON" - "${TASK_GPU_PIDS[@]}" <<'PY'
import os
import signal
import sys

groups = set()
for value in sys.argv[1:]:
    pid = int(value)
    if pid == 1:
        raise SystemExit("refusing PID 1")
    groups.add(os.getpgid(pid))
if 1 in groups:
    raise SystemExit("refusing process group 1")
for group in sorted(groups):
    print(f"SIGTERM process_group={group}", flush=True)
    os.killpg(group, signal.SIGTERM)
PY
  sleep 5
done

mapfile -t TASK_REMAINING_GPU_PIDS < <(gpu_pids)
if (( ${#TASK_REMAINING_GPU_PIDS[@]} )); then
  TASK_ALL_OWNED=1
  for TASK_PID in "${TASK_REMAINING_GPU_PIDS[@]}"; do
    TASK_CMD="$(tr '\0' ' ' <"/proc/$TASK_PID/cmdline" 2>/dev/null || true)"
    echo "remaining_gpu_pid=$TASK_PID cmd=$TASK_CMD"
    if [[ "$TASK_CMD" != *train_compositional.py* ]] || \
       { [[ "$TASK_CMD" != *'--arm product_code'* ]] && \
         [[ "$TASK_CMD" != *'--arm residual_subspace_experts'* ]]; }; then
      TASK_ALL_OWNED=0
    fi
  done
  test "$TASK_ALL_OWNED" -eq 1
  kill -9 "${TASK_REMAINING_GPU_PIDS[@]}" 2>/dev/null || true
  sleep 5
fi

echo '=== stop any remaining exact experiment controllers, never PID 1 ==='
"$TASK_PYTHON" - "$TASK_MAIN_LOG_DIR" "$TASK_RETRY_LOG_DIR" <<'PY'
import os
import signal
import sys
from pathlib import Path

needles = tuple(sys.argv[1:])
self_pid = os.getpid()
ancestors = {self_pid}
ancestor = os.getppid()
while ancestor > 0 and ancestor not in ancestors:
    ancestors.add(ancestor)
    try:
        with open(f"/proc/{ancestor}/status") as handle:
            parent_line = next(line for line in handle if line.startswith("PPid:"))
        ancestor = int(parent_line.split()[1])
    except (FileNotFoundError, PermissionError, ProcessLookupError, StopIteration):
        break
groups = set()
matches = []
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    pid = int(entry.name)
    if pid == 1 or pid in ancestors:
        continue
    try:
        command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", "replace"
        )
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if any(needle in command for needle in needles):
        group = os.getpgid(pid)
        if group == 1:
            raise SystemExit(f"refusing process group 1 for pid={pid}")
        matches.append((pid, group, command))
        groups.add(group)
for pid, group, command in matches:
    print(f"controller_pid={pid} pgid={group} cmd={command}", flush=True)
for group in sorted(groups):
    os.killpg(group, signal.SIGTERM)
PY
sleep 10

echo '=== verify all GPUs and requested controllers are stopped ==='
mapfile -t TASK_FINAL_GPU_PIDS < <(gpu_pids)
test "${#TASK_FINAL_GPU_PIDS[@]}" -eq 0
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,power.draw \
  --format=csv,noheader
"$TASK_PYTHON" - "$TASK_MAIN_LOG_DIR" "$TASK_RETRY_LOG_DIR" <<'PY'
import os
import sys
from pathlib import Path

needles = tuple(sys.argv[1:])
ancestors = {os.getpid()}
ancestor = os.getppid()
while ancestor > 0 and ancestor not in ancestors:
    ancestors.add(ancestor)
    try:
        with open(f"/proc/{ancestor}/status") as handle:
            parent_line = next(line for line in handle if line.startswith("PPid:"))
        ancestor = int(parent_line.split()[1])
    except (FileNotFoundError, PermissionError, ProcessLookupError, StopIteration):
        break
remaining = []
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit() or int(entry.name) in ancestors:
        continue
    try:
        command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", "replace"
        )
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    if "run_experiments.py" in command and any(needle in command for needle in needles):
        remaining.append((entry.name, command))
if remaining:
    raise SystemExit(f"requested experiment controllers remain: {remaining}")
print("REQUESTED_EXPERIMENT_CONTROLLERS_STOPPED")
PY
echo 'FINAL_RSE_HASHED_RUNS_STOPPED_AND_ALL_GPUS_FREE'

echo '=== remove only the two requested experiment outputs and their logs ==='
for TASK_PATH in \
    "$TASK_RSE_OUTPUT" "$TASK_HASHED_OUTPUT" \
    "$TASK_MAIN_LOG_DIR" "$TASK_RETRY_LOG_DIR" \
    "$TASK_RSE_WANDB" "$TASK_HASHED_WANDB"; do
  if [[ -e "$TASK_PATH" || -L "$TASK_PATH" ]]; then
    echo "DELETE: $TASK_PATH"
    rm -rf -- "$TASK_PATH"
  else
    echo "ABSENT: $TASK_PATH"
  fi
done

if [[ -L "$TASK_PROJECT/wandb/latest-run" ]]; then
  TASK_LATEST_TARGET="$(readlink -f "$TASK_PROJECT/wandb/latest-run" 2>/dev/null || true)"
  if [[ "$TASK_LATEST_TARGET" = "$TASK_RSE_WANDB" || \
        "$TASK_LATEST_TARGET" = "$TASK_HASHED_WANDB" || \
        -z "$TASK_LATEST_TARGET" ]]; then
    rm -f -- "$TASK_PROJECT/wandb/latest-run"
  fi
fi

echo '=== remove Hugging Face dataset cache plus project data cache-* and tmp* ==='
rm -rf -- /mnt/local/.cache/huggingface/datasets
if [[ -d "$TASK_DATA_ROOT" ]]; then
  find "$TASK_DATA_ROOT" -type f \( -name 'cache-*' -o -name 'tmp*' \) -print -delete
  find "$TASK_DATA_ROOT" -depth -type d -name 'tmp*' -print -exec rm -rf -- {} +
fi

echo '=== final cleanup verification ==='
test ! -e "$TASK_RSE_OUTPUT"
test ! -e "$TASK_HASHED_OUTPUT"
test ! -e "$TASK_MAIN_LOG_DIR"
test ! -e "$TASK_RETRY_LOG_DIR"
test ! -e "$TASK_RSE_WANDB"
test ! -e "$TASK_HASHED_WANDB"
test ! -e /mnt/local/.cache/huggingface/datasets
TASK_CACHE_FILES=0
if [[ -d "$TASK_DATA_ROOT" ]]; then
  TASK_CACHE_FILES="$(find "$TASK_DATA_ROOT" -type f \( -name 'cache-*' -o -name 'tmp*' \) | wc -l)"
fi
test "$TASK_CACHE_FILES" -eq 0
mapfile -t TASK_POST_CLEAN_GPU_PIDS < <(gpu_pids)
test "${#TASK_POST_CLEAN_GPU_PIDS[@]}" -eq 0
nvidia-smi
echo "remaining_project_dataset_cache_or_tmp_files=$TASK_CACHE_FILES"
echo 'TH2 FINAL RSE HASHED OUTPUT AND DATASET CACHE CLEANUP COMPLETE; ALL GPUS FREE'
