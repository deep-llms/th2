#1 +60+a
#th2-q359-A-time-match10800-20261010-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
TASK_ROOT=/mnt/local/_outputs/deep-llms_th2/q359-A-time-match10800-seed1042-20261010-a01
TASK_SOURCE=/mnt/local/_outputs/deep-llms_th2/q359-proxy-10k-seed1042-20261009-a01/supervised/run
TASK_SESSION=q359-A-time-match10800-seed1042-20261010-a01
test ! -e "$TASK_ROOT"
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then exit 1; fi
mkdir -p "$TASK_ROOT"
tmux new-session -d -s "$TASK_SESSION" "exec bash scripts/launch_proxy_time_match.sh '$TASK_ROOT' thiennh-p6-q359-worker-0 '$TASK_SOURCE' >'$TASK_ROOT/supervisor.log' 2>&1"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 45
cat "$TASK_ROOT/supervisor.log"
if [ "$(tmux display-message -p -t "$TASK_SESSION" '#{pane_dead}')" = 1 ]; then
  if [ -f "$TASK_ROOT/retention-test.log" ]; then tail -70 "$TASK_ROOT/retention-test.log"; fi
  /mnt/local/conda-py311/envs/attention_bench/bin/python3.11 - "$TASK_ROOT" <<'PYREMOTE'
import json,sys
from pathlib import Path
r=json.loads((Path(sys.argv[1])/'supervised/supervisor.json').read_text())
assert r['training_status']=='ok' and r['training_returncode']==0 and r['burn']['collective_progress_verified'],r
print('TIME_MATCH_AND_BURN_COMPLETE',flush=True)
PYREMOTE
else
  echo 'TIME_MATCH_SUPERVISOR_ACTIVE; inspect preflight and live progress.'
fi
