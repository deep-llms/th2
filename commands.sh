#1 +60+a
#th2-q359-proxy-resume20k-seed1042-20261011-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
TASK_ROOT=/mnt/local/_outputs/deep-llms_th2/q359-proxy-resume20k-seed1042-20261011-a01
TASK_SESSION=q359-proxy-resume20k-seed1042-20261011-a01
TASK_SOURCE=/mnt/local/_outputs/deep-llms_th2/q359-proxy-10k-seed1042-20261009-a01/supervised/run
TASK_STOP=/mnt/local/_outputs/deep-llms_th2/q359-stop-seed42-20261011-a01/stop.json
test -s "$TASK_STOP"
test ! -e "$TASK_ROOT"
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then exit 1; fi
mkdir -p "$TASK_ROOT"
tmux new-session -d -s "$TASK_SESSION" "exec bash scripts/launch_proxy_resume_pair.sh '$TASK_ROOT' thiennh-p6-q359-worker-0 '$TASK_SOURCE' '$TASK_STOP' >'$TASK_ROOT/supervisor.log' 2>&1"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 45
tail -45 "$TASK_ROOT/supervisor.log"
tmux list-panes -t "$TASK_SESSION" -F '#{pane_dead} #{pane_pid}'
