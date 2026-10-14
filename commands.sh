#1 +60+a
#th2-q359-p7-teacher-probe-20261015-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
TASK_ROOT=/mnt/local/_outputs/deep-llms_th2/q359-p7-teacher-probe-20261015-a01
TASK_SESSION=q359-p7-teacher-probe-20261015-a01
TASK_PREDECESSOR=/mnt/local/_outputs/deep-llms_th2/q359-p7-consumer4-20261015-a01
test ! -e "$TASK_ROOT"
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then
  echo 'REFUSE: teacher probe session already exists' >&2
  exit 1
fi
mkdir "$TASK_ROOT"
printf -v TASK_COMMAND 'bash scripts/launch_p7_teacher_probe.sh %q %q %q >%q 2>&1' \
  "$TASK_ROOT" thiennh-p6-q359-worker-0 "$TASK_PREDECESSOR" "$TASK_ROOT/supervisor.log"
tmux new-session -d -s "$TASK_SESSION" "$TASK_COMMAND"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 5
tmux list-panes -t "$TASK_SESSION" -F '#{pane_dead} #{pane_pid} #{pane_current_command}'
tail -n 15 "$TASK_ROOT/supervisor.log"
