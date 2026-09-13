#!/usr/bin/env bash
# Arm a persistent handoff; do not stop the current CPU preparation or burns.
set -euo pipefail
test "$(hostname)" = thiennh-p6-8mgy-worker-0
cd /mnt/local/deep-llms_th2
source /mnt/local/conda-py311/etc/profile.d/conda.sh
conda activate train_env
test "$CONDA_DEFAULT_ENV" = train_env
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false WANDB_MODE=offline PYTHONUNBUFFERED=1
TASK_SESSION=ccm_pilot_overnight_20260913_a01
TASK_LOG=/mnt/local/_outputs/deep-llms_th2/ccm_pilot_seed17_20260913_a01.handoff.log
TASK_OUTPUT=/mnt/local/_outputs/deep-llms_th2/ccm_pilot_seed17_20260913_a01
TASK_PREP_LOG=/mnt/local/deep-llms_th2/_run_log_/_run-2026-09-13_01-36-41-th2-ccm-prepare-full-pilot-20260913-a01.log
test -s "$TASK_PREP_LOG"
test ! -e "$TASK_OUTPUT" && test ! -L "$TASK_OUTPUT"
test ! -e "$TASK_LOG" && test ! -L "$TASK_LOG"
command -v tmux >/dev/null
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then
    echo 'REFUSE: handoff session already exists' >&2
    exit 1
fi
/usr/bin/python3 scripts/pilot_gpu_ops.py inspect --gpus 0 1 2 3 4 5 6 7
# New tmux shells inherit the server's environment, not necessarily this shell's
# conda activation. Activate explicitly inside the persistent child too.
printf -v TASK_CMD 'cd /mnt/local/deep-llms_th2 && source /mnt/local/conda-py311/etc/profile.d/conda.sh && conda activate train_env && exec /usr/bin/python3 -u scripts/pilot_overnight.py --prep-log %q >%q 2>&1' "$TASK_PREP_LOG" "$TASK_LOG"
tmux new-session -d -s "$TASK_SESSION" "bash -c $(printf '%q' "$TASK_CMD")"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 30
test "$(tmux display-message -p -t "$TASK_SESSION" '#{pane_dead}')" = 0
/usr/bin/python3 - "$TASK_OUTPUT/status.json" <<'PY'
import json, sys
from datetime import datetime, timezone
s = json.load(open(sys.argv[1]))
assert s['stage'] == 'waiting_for_preparation', s
assert s['event'] == 'waiting', s
assert (datetime.now(timezone.utc)-datetime.fromisoformat(s['time'])).total_seconds() < 120
print('CCM_PILOT_OVERNIGHT_HANDOFF_ARMED', s)
PY
tail -n 5 "$TASK_LOG"
