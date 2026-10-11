#1 +60+a
#th2-q359-proxy-10k-seed42-20261011-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
TASK_ROOT=/mnt/local/_outputs/deep-llms_th2/q359-proxy-10k-seed42-20261011-a01
TASK_SESSION=q359-proxy-10k-seed42-20261011-a01
TASK_PARENT=/mnt/local/_outputs/deep-llms_th2/q359-p6-cooldown-20261011-a01
test ! -e "$TASK_ROOT"
test -d "$TASK_PARENT"
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then exit 1; fi
df -h /mnt/local/_outputs/deep-llms_th2 /mnt/local/_data/deep-llms_th2
mkdir -p "$TASK_ROOT"
tmux new-session -d -s "$TASK_SESSION" "exec bash scripts/launch_proxy_10k_fresh.sh '$TASK_ROOT' thiennh-p6-q359-worker-0 42 '$TASK_PARENT' >'$TASK_ROOT/supervisor.log' 2>&1"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 45
tail -45 "$TASK_ROOT/supervisor.log"
test "$(tmux display-message -p -t "$TASK_SESSION" '#{pane_dead}')" = 0
echo 'SEED42_REPLICATION_SUPERVISOR_ALIVE'
tmux list-panes -t "$TASK_SESSION" -F '#{pane_dead} #{pane_pid}'
