#1 +60+a
#th2-q359-p6-compile-20261011-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
TASK_ROOT=/mnt/local/_outputs/deep-llms_th2/q359-p6-compile-20261011-a01
TASK_SESSION=q359-p6-compile-20261011-a01
test ! -e "$TASK_ROOT"
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then exit 1; fi
mkdir -p "$TASK_ROOT"
tmux new-session -d -s "$TASK_SESSION" "exec bash scripts/launch_p6_compile_trial.sh '$TASK_ROOT' thiennh-p6-q359-worker-0 >'$TASK_ROOT/supervisor.log' 2>&1"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 45
tail -45 "$TASK_ROOT/supervisor.log"
if [ "$(tmux display-message -p -t "$TASK_SESSION" '#{pane_dead}')" = 1 ]; then
  if [ -f "$TASK_ROOT/cpu-tests.log" ]; then tail -65 "$TASK_ROOT/cpu-tests.log"; fi
  /mnt/local/conda-py311/envs/eval_fa4/bin/python3.11 - "$TASK_ROOT" <<'PYREMOTE'
import json,sys
from pathlib import Path
r=json.loads((Path(sys.argv[1])/'supervised/supervisor.json').read_text())
assert r['training_status']=='ok' and r['training_returncode']==0 and r['burn']['collective_progress_verified'],r
print('COMPILE_STUDY_AND_BURN_COMPLETE',flush=True)
PYREMOTE
else
  echo 'COMPILE_SUPERVISOR_ACTIVE; awaiting CPU preflight or running queue.'
fi
