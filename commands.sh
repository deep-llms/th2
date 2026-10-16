#1 +15+a
#th2-q359-p7-tuning-eval-20261016-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
TASK_ROOT=/mnt/local/_outputs/deep-llms_th2/q359-p7-tuning-eval-20261016-a01
TASK_SOURCE=/mnt/local/_outputs/deep-llms_th2/q359-p7-priority15-20261015-a01
TASK_REFERENCE=/mnt/local/_outputs/deep-llms_th2/q359-proxy12-eval-remaining27-20261014-a01
TASK_EARLIER=/mnt/local/_outputs/deep-llms_th2/q359-p7-tuning22-20261015-a03
TASK_SESSION=q359-p7-tuning-eval-20261016-a01
test ! -e "$TASK_ROOT"
test ! -e "$TASK_ROOT.launch.log"
test -s "$TASK_SOURCE/user-stop-for-evaluation.json"
cat "$TASK_SOURCE.stop-for-eval.log"
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then exit 1; fi
tmux new-session -d -s "$TASK_SESSION" "cd /mnt/local/deep-llms_th2 && exec bash scripts/launch_p7_tuning_eval.sh $TASK_ROOT $TASK_SOURCE $TASK_REFERENCE thiennh-p6-q359-worker-0 $TASK_EARLIER > $TASK_ROOT.launch.log 2>&1"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 5
tail -c 5000 "$TASK_ROOT.launch.log"
