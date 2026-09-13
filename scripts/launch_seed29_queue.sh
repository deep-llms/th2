#!/usr/bin/env bash
set -euo pipefail
test "$(hostname)" = thiennh-p6-8mgy-worker-0
cd /mnt/local/deep-llms_th2
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate train_env
test "$CONDA_DEFAULT_ENV" = train_env
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline PYTHONUNBUFFERED=1
TASK_SESSION=ccm_seed29_common_20260913_a01
TASK_OUTPUT=/mnt/local/_outputs/deep-llms_th2/ccm_pilot_seed29_20260913_a01
TASK_LOG=/mnt/local/_outputs/deep-llms_th2/ccm_pilot_seed29_20260913_a01.handoff.log
test ! -e "$TASK_OUTPUT" && test ! -L "$TASK_OUTPUT"
test ! -e "$TASK_LOG" && test ! -L "$TASK_LOG"
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then
    echo 'REFUSE: Seed-29 queue already exists' >&2
    exit 1
fi
/usr/bin/python3 -c 'import sys; sys.path.insert(0, "scripts"); from pilot_gpu_ops import preflight; preflight()'
# Do not inspect/stop burns here: current GPUs may be training or evaluating.
printf -v TASK_CMD 'cd /mnt/local/deep-llms_th2 && source /mnt/local/conda-py311/etc/profile.d/conda.sh && conda activate train_env && exec /usr/bin/python3 -u scripts/queue_seed29.py >%q 2>&1' "$TASK_LOG"
tmux new-session -d -s "$TASK_SESSION" "bash -c $(printf '%q' "$TASK_CMD")"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 30
test "$(tmux display-message -p -t "$TASK_SESSION" '#{pane_dead}')" = 0
/usr/bin/python3 - "$TASK_OUTPUT/status.json" <<'PY'
import json, sys
from datetime import datetime, timezone
s = json.load(open(sys.argv[1]))
assert s['stage'] != 'failed', s
assert (datetime.now(timezone.utc)-datetime.fromisoformat(s['time'])).total_seconds() < 120
print('CCM_SEED29_QUEUE_ARMED', s)
PY
