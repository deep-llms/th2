#1 +60+a
#th2-78gg-deep-kv-2500-launch-20260928-a01
set -euo pipefail
date -u
hostname
test "$(hostname)" = thiennh-p6-78gg-worker-0
TASK_ROOT=/mnt/local/_outputs/@PROJECT@/deep-kv-2500-20260928-a01
TASK_INSPECTION=/mnt/local/_outputs/@PROJECT@/deep-kv-2500-preflight-20260928-a01/gpu_inspection.json
TASK_SESSION=deep-kv-2500-20260928-a01
test -s "$TASK_INSPECTION"
test ! -e "$TASK_ROOT"
test ! -e "${TASK_ROOT}.log"
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then
  echo 'REFUSE: training supervisor session already exists' >&2
  exit 1
fi
printf -v TASK_CMD 'cd %q && exec bash scripts/launch_deep_kv_b200.sh %q %q >%q 2>&1' \
  "$PWD" "$TASK_ROOT" "$TASK_INSPECTION" "${TASK_ROOT}.log"
tmux new-session -d -s "$TASK_SESSION" "$TASK_CMD"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 45
test "$(tmux display-message -p -t "$TASK_SESSION" '#{pane_dead}')" = 0
tail -n 65 "${TASK_ROOT}.log"
printf '%s\n' 'DETACHED_PIPELINE_LIVE: verified crash-to-burn rehearsal, then real four-arm queue'
