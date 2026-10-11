#1 +60+a
#th2-q359-stop-seed42-20261011-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
TASK_ROOT=/mnt/local/_outputs/deep-llms_th2/q359-stop-seed42-20261011-a01
TASK_OLD=/mnt/local/_outputs/deep-llms_th2/q359-proxy-10k-seed42-20261011-a01
TASK_SESSION=q359-stop-seed42-20261011-a01
test ! -e "$TASK_ROOT"
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then exit 1; fi
mkdir -p "$TASK_ROOT"
tmux new-session -d -s "$TASK_SESSION" "exec /mnt/local/conda-py311/envs/eval_fa4/bin/python3.11 -u -m scripts.stop_proxy_followup --root '$TASK_OLD' --session q359-proxy-10k-seed42-20261011-a01 --host thiennh-p6-q359-worker-0 --burn-session q359-proxy-10k-seed42-20261011-a01-final-burn --output '$TASK_ROOT/stop.json' >'$TASK_ROOT/stop.log' 2>&1"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 45
tail -30 "$TASK_ROOT/stop.log"
tmux list-panes -t "$TASK_SESSION" -F '#{pane_dead} #{pane_pid}'
