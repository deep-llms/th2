#1 +60+a
#th2-q359-proxy-12layer-screen22-20261013-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
TASK_ROOT=/mnt/local/_outputs/deep-llms_th2/q359-proxy-12layer-screen22-20261013-a01
TASK_SESSION=q359-proxy-12layer-screen22-20261013-a01
test ! -e "$TASK_ROOT"
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then exit 1; fi
mkdir -p "$TASK_ROOT"
tmux new-session -d -s "$TASK_SESSION" "exec bash scripts/launch_proxy_12layer.sh '$TASK_ROOT' thiennh-p6-q359-worker-0 >'$TASK_ROOT/supervisor.log' 2>&1"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 45
tail -55 "$TASK_ROOT/supervisor.log"
if [ "$(tmux display-message -p -t "$TASK_SESSION" '#{pane_dead}')" = 1 ]; then
  echo 'SCREEN22_LAUNCHER_EXITED; inspect supervisor receipt before taking action.'
  test -f "$TASK_ROOT/supervised/supervisor.json"
  cat "$TASK_ROOT/supervised/supervisor.json"
  exit 1
fi
echo 'SCREEN22_SUPERVISOR_ACTIVE; numerical and Trainer gates precede production.'
