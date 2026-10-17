#1 +60+a
#th2-q359-ce7454-followup-20261017-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-q359-worker-0
test "$PWD" = /mnt/local/deep-llms_th2
TASK_ROOT=/mnt/local/_outputs/deep-llms_th2/q359-ce7454-followup-20261017-a01
TASK_PY=/mnt/local/conda-py311/envs/eval_fa4/bin/python
TASK_SESSION=ce7454_followup_20261017_a01
test ! -e "$TASK_ROOT"
test ! -e "$TASK_ROOT.log"
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then
  echo 'REFUSE: CE7454 session already exists' >&2
  exit 1
fi
printf -v TASK_CMD 'exec %q -u %q --output %q --reuse-verified-cache --study followup --previous-output /mnt/local/_outputs/deep-llms_th2/q359-ce7454-scratch-20261016-a04 >%q 2>&1' \
  "$TASK_PY" "$PWD/ce7454/src/launch_b200_study.py" "$TASK_ROOT" "$TASK_ROOT.log"
tmux new-session -d -s "$TASK_SESSION" "$TASK_CMD"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
echo "CE7454_LAUNCH_SUBMITTED $TASK_ROOT"
sleep 10
tail -n 25 "$TASK_ROOT.log"
