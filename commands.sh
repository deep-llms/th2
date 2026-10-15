#1 +60+a
#th2-q359-p7-exploration8-20261015-a02
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
date -u
printf 'SOURCE_IMPLEMENTATION 6ce50b3\n'
nvidia-smi
TASK_ROOT=/mnt/local/_outputs/deep-llms_th2/q359-p7-exploration8-20261015-a02
TASK_SESSION=q359-p7-exploration8-20261015-a02
TASK_LOG=/mnt/local/_outputs/deep-llms_th2/q359-p7-exploration8-20261015-a02.launch.log
TASK_PREDECESSOR=/mnt/local/_outputs/deep-llms_th2/q359-p7-target-integrated-20261015-a01
TASK_REFERENCE=/mnt/local/_outputs/deep-llms_th2/q359-proxy12-eval-remaining27-20261014-a01
TASK_ROWS=/mnt/local/_outputs/deep-llms_th2/q359-p7-teacher-probe-20261015-a01/rows.json
test ! -e "$TASK_ROOT"
test ! -e "$TASK_LOG"
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then
  echo 'REFUSE: exploration session already exists' >&2
  exit 1
fi
printf -v TASK_COMMAND 'bash scripts/launch_p7_exploration.sh %q %q %q %q %q >%q 2>&1' \
  "$TASK_ROOT" thiennh-p6-q359-worker-0 "$TASK_PREDECESSOR" "$TASK_REFERENCE" "$TASK_ROWS" "$TASK_LOG"
tmux new-session -d -s "$TASK_SESSION" "$TASK_COMMAND"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 5
tmux list-panes -t "$TASK_SESSION" -F '#{pane_dead} #{pane_pid} #{pane_current_command}'
tail -n 25 "$TASK_LOG"
