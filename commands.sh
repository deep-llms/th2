#1 +60+a
#th2-check-pure-local-live-progress-20260825-2106
set -euo pipefail

TASK_OUTPUT_BASE=/mnt/local/_outputs/@PROJECT@
TASK_OUTPUT_DIR="$TASK_OUTPUT_BASE/pure_local_tied_g16_r128"
TASK_LOG_DIR="$TASK_OUTPUT_BASE/logs/pure_local_tied_g16_r128_20260825"
TASK_TRAIN_LOG="$TASK_LOG_DIR/pure_local_tied_g16_r128.log"
TASK_EXPERIMENT_LOG="$TASK_LOG_DIR/experiments.log"
TASK_EVAL_LOG="$TASK_OUTPUT_BASE/eval_parallel_pure_local_tied_g16_r128_10k_20260825.log"
TASK_FINETUNE_DIR="$TASK_OUTPUT_BASE/finetune_pure_local_tied_g16_r128_10k_20260825"
TASK_PYTHON=/mnt/local/conda-py311/envs/sparse_emb/bin/python3.11

echo '=== PureLocal live progress (read-only) ==='
date -u
hostname
test -x "$TASK_PYTHON"
test -d "$TASK_OUTPUT_DIR"
test -s "$TASK_TRAIN_LOG"
test -s "$TASK_EXPERIMENT_LOG"

echo '=== GPU state and process stage ==='
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
mapfile -t TASK_GPU_PIDS < <(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | awk 'NF {gsub(/[[:space:]]/, "", $0); print}' | sort -nu
)
echo "gpu_process_count=${#TASK_GPU_PIDS[@]}"
for TASK_PID in "${TASK_GPU_PIDS[@]}"; do
    if test -r "/proc/$TASK_PID/cmdline"; then
        TASK_CMDLINE=$(tr '\0' ' ' < "/proc/$TASK_PID/cmdline")
        echo "gpu_pid=$TASK_PID cmd=$TASK_CMDLINE"
    else
        echo "gpu_pid=$TASK_PID cmd=<exited-during-snapshot>"
    fi
done

echo '=== checkpoints ==='
find "$TASK_OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' \
    -printf '%f %TY-%Tm-%TdT%TH:%TM:%TSZ\n' | sort -V | tail -n 8

echo '=== derive latest step, rate, and training ETA ==='
"$TASK_PYTHON" - "$TASK_TRAIN_LOG" <<'PY'
import datetime as dt
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
with path.open("rb") as handle:
    handle.seek(0, 2)
    size = handle.tell()
    handle.seek(max(0, size - 8_000_000))
    text = handle.read().decode("utf-8", errors="replace").replace("\r", "\n")

matches = re.findall(
    r"(\d+)/33339\s+\[[^\n]*?,\s*([0-9]+(?:\.[0-9]+)?)s/it\]",
    text,
)
if not matches:
    raise SystemExit("ERROR: no recent tqdm progress record found")
step, seconds_per_step = int(matches[-1][0]), float(matches[-1][1])
target = 10_000
now = dt.datetime.now(dt.timezone.utc)
remaining_seconds = max(0, target - step) * seconds_per_step
eta = now + dt.timedelta(seconds=remaining_seconds)
print(f"live_step={step}")
print(f"target_step={target}")
print(f"seconds_per_step={seconds_per_step:.3f}")
print(f"training_remaining_seconds={remaining_seconds:.0f}")
print(f"training_eta_utc={eta:%Y-%m-%d %H:%M:%S UTC}")

metric_rows = re.findall(r"\{'loss':[^\n]+", text)
if metric_rows:
    print(f"latest_metric={metric_rows[-1]}")
PY

echo '=== workflow artifact/stage indicators ==='
if test -d "$TASK_OUTPUT_DIR/checkpoint-10000"; then
    echo 'checkpoint_10000=present'
else
    echo 'checkpoint_10000=absent'
fi
if test -s "$TASK_EVAL_LOG"; then
    echo "eval_log=present bytes=$(stat -c %s "$TASK_EVAL_LOG")"
    tail -n 20 "$TASK_EVAL_LOG"
else
    echo 'eval_log=absent'
fi
if test -d "$TASK_FINETUNE_DIR"; then
    echo "finetune_json_count=$(find "$TASK_FINETUNE_DIR" -maxdepth 1 -type f -name '*.json' | wc -l)"
else
    echo 'finetune_dir=absent'
fi

echo '=== latest experiment state ==='
tail -n 20 "$TASK_EXPERIMENT_LOG"

echo 'TH2 PURE-LOCAL LIVE PROGRESS CHECK OK'
