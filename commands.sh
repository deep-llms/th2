#1 +180+a
#th2-q359-p7-completed-eval-20261015-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
TASK_ROOT=/mnt/local/_outputs/deep-llms_th2/q359-p7-completed-eval-20261015-a01
TASK_SOURCE=/mnt/local/_outputs/deep-llms_th2/q359-p7-exploration8-20261015-a03
TASK_REFERENCE=/mnt/local/_outputs/deep-llms_th2/q359-proxy12-eval-remaining27-20261014-a01
TASK_SESSION=q359-p7-completed-eval-20261015-a01
TASK_LOG=/mnt/local/_outputs/deep-llms_th2/q359-p7-completed-eval-20261015-a01.launch.log
test ! -e "$TASK_ROOT"
test ! -e "$TASK_LOG"
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then
 echo 'REFUSE: evaluation session already exists' >&2
 exit 1
fi
printf -v TASK_COMMAND 'exec bash scripts/launch_p7_completed_eval.sh %q %q %q %q >%q 2>&1' \
 "$TASK_ROOT" "$TASK_SOURCE" "$TASK_REFERENCE" thiennh-p6-q359-worker-0 "$TASK_LOG"
tmux new-session -d -s "$TASK_SESSION" "$TASK_COMMAND"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 150
tmux list-panes -t "$TASK_SESSION" -F '#{pane_dead} #{pane_pid} #{pane_current_command}'
tail -n 100 "$TASK_LOG"
