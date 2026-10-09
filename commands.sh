#1 +60+a
#th2-q359-fa4-runtime-smoke-20261009-a01
set -euo pipefail
TASK_ROOT=/mnt/local/_outputs/deep-llms_th2/q359-fa4-runtime-smoke-20261009-a01
TASK_SESSION=q359-fa4-runtime-smoke-20261009-a01
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
test ! -e "$TASK_ROOT"
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then exit 1; fi
mkdir -p "$TASK_ROOT"
tmux new-session -d -s "$TASK_SESSION" "exec bash scripts/launch_fa4_runtime_smoke.sh '$TASK_ROOT' thiennh-p6-q359-worker-0 >'$TASK_ROOT/supervisor.log' 2>&1"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
for task_poll in $(seq 1 60); do
  sleep 10
  if [ "$(tmux display-message -p -t "$TASK_SESSION" '#{pane_dead}')" = 1 ]; then
    cat "$TASK_ROOT/supervisor.log"
    test ! -f "$TASK_ROOT/supervised/run/fa4-runtime-smoke.log" || cat "$TASK_ROOT/supervised/run/fa4-runtime-smoke.log"
    test ! -f "$TASK_ROOT/supervised/supervisor.json" || cat "$TASK_ROOT/supervised/supervisor.json"
    exit "$(tmux display-message -p -t "$TASK_SESSION" '#{pane_dead_status}')"
  fi
  if [ "$((task_poll % 3))" = 0 ]; then
    tail -n 5 "$TASK_ROOT/supervisor.log"
  fi
done
echo 'Smoke supervisor remains active; inspect result and burn receipts before declaring completion.'
