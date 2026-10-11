#1 +60+a
#th2-q359-checkpoint-ppl-20261012-a02
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
TASK_ROOT=/mnt/local/_outputs/deep-llms_th2/q359-checkpoint-ppl-20261012-a02
TASK_SESSION=q359-checkpoint-ppl-20261012-a02
TASK_SOURCE=/mnt/local/_outputs/deep-llms_th2/q359-proxy-resume20k-seed1042-20261011-a01/supervised/run
test ! -e "$TASK_ROOT"
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then exit 1; fi
mkdir -p "$TASK_ROOT"
tmux new-session -d -s "$TASK_SESSION" "exec bash scripts/launch_checkpoint_ppl.sh '$TASK_ROOT' thiennh-p6-q359-worker-0 '$TASK_SOURCE' >'$TASK_ROOT/supervisor.log' 2>&1"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 45
tail -35 "$TASK_ROOT/supervisor.log"
tmux list-panes -t "$TASK_SESSION" -F '#{pane_dead} #{pane_pid}'
