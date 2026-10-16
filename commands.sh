#1 +15+a
#th2-q359-p7-stop-for-eval-20261016-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
TASK_ROOT=/mnt/local/_outputs/deep-llms_th2/q359-p7-priority15-20261015-a01
TASK_SESSION=q359-p7-stop-for-eval-20261016-a01
test ! -e "$TASK_ROOT/user-stop-for-evaluation.json"
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then exit 1; fi
tmux new-session -d -s "$TASK_SESSION" "cd /mnt/local/deep-llms_th2 && exec /mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u -m scripts.stop_proxy_followup --root $TASK_ROOT --session q359-p7-priority15-20261015-a01 --host thiennh-p6-q359-worker-0 --burn-session q359-p7-priority15-20261015-a01-final-burn --output $TASK_ROOT/user-stop-for-evaluation.json > $TASK_ROOT.stop-for-eval.log 2>&1"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 5
cat "$TASK_ROOT.stop-for-eval.log"
