#1 +60+a
#th2-q359-paired-parallel-repair-20261013-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
TASK_ROOT=/mnt/local/_outputs/deep-llms_th2/q359-paired-parallel-repair-20261013-a01
TASK_SOURCE=/mnt/local/_outputs/deep-llms_th2/q359-expanded-single-seed42-20261013-a01
TASK_SESSION=q359-paired-parallel-repair-20261013-a01
test ! -e "$TASK_ROOT"
test ! -e "${TASK_ROOT}-launch.log"
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then exit 1; fi
printf -v TASK_CMD 'exec bash scripts/launch_paired_restart.sh %q %q %q >%q 2>&1' \
  "$TASK_SOURCE" "$TASK_ROOT" thiennh-p6-q359-worker-0 "${TASK_ROOT}-launch.log"
tmux new-session -d -s "$TASK_SESSION" "$TASK_CMD"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 45
tail -n 60 "${TASK_ROOT}-launch.log"
if [ -f "$TASK_ROOT/supervised/supervisor.json" ]; then cat "$TASK_ROOT/supervised/supervisor.json"; fi
test "$(tmux display-message -p -t "$TASK_SESSION" '#{pane_dead}')" = 0
