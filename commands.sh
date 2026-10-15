#1 +60+a
#th2-q359-p7-kernel-trial-20261015-a02
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
TASK_ROOT=/mnt/local/_outputs/deep-llms_th2/q359-p7-kernel-trial-20261015-a02
TASK_SESSION=q359-p7-kernel-trial-20261015-a02
TASK_PREDECESSOR=/mnt/local/_outputs/deep-llms_th2/q359-p7-entries-integrated-20261015-a01
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 - <<'CHECK'
from pathlib import Path
import json
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-p7-kernel-trial-20261015-a01/supervised')
s,r,b=(json.loads((root/p).read_text()) for p in ('supervisor.json','run/run.json','burn-verified.json'))
assert s['training_status']=='failed' and s.get('finished_at')
assert r['status']=='failed' and r['error']=='numerical-gate exited with code 1'
assert len(r['jobs'])==1 and r['jobs'][0]['name']=='numerical-gate'
assert b['collective_progress_verified'] and s['burn']['collective_progress_verified']
assert b['session']==s['burn']['session']
assert not Path('/mnt/local/_gpu_guard/DISABLED').exists()
print('FAILED_PRIMITIVE_GATE_PRESERVED_AND_BURN_HANDOFF_VERIFIED',flush=True)
CHECK
test ! -e "$TASK_ROOT"
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then
  echo 'REFUSE: speed profile session already exists' >&2
  exit 1
fi
mkdir "$TASK_ROOT"
printf -v TASK_COMMAND 'bash scripts/launch_p7_speed_study.sh %q %q %q kernel-study >%q 2>&1' \
  "$TASK_ROOT" thiennh-p6-q359-worker-0 "$TASK_PREDECESSOR" "$TASK_ROOT/supervisor.log"
tmux new-session -d -s "$TASK_SESSION" "$TASK_COMMAND"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 5
tmux list-panes -t "$TASK_SESSION" -F '#{pane_dead} #{pane_pid} #{pane_current_command}'
tail -n 15 "$TASK_ROOT/supervisor.log"
