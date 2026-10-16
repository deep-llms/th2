#1 +15+a
#th2-q359-teacher-first-pilot-20261016-a02
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
TASK_ROOT=/mnt/local/_outputs/deep-llms_th2/q359-teacher-first-20261016-a02
TASK_PREVIOUS=/mnt/local/_outputs/deep-llms_th2/q359-teacher-first-20261016-a01
TASK_SESSION=q359-teacher-first-20261016-a02
test ! -e "$TASK_ROOT"
test ! -e "$TASK_ROOT.launch.log"
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then exit 1; fi
tmux new-session -d -s "$TASK_SESSION" "cd /mnt/local/deep-llms_th2 && exec bash scripts/launch_teacher_first.sh $TASK_ROOT $TASK_PREVIOUS > $TASK_ROOT.launch.log 2>&1"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 5
tail -c 5000 "$TASK_ROOT.launch.log"
