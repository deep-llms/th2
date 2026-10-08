#1 +60+a
#th2-tjx3-recover-A-P6iso-10k-launch-20261008-a02
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
TASK_ROOT=/mnt/local/_outputs/@PROJECT@/proxy-A-P6iso-10k-seed1042-20261008-a01
TASK_SESSION=tjx3-proxy-A-P6iso-10k-seed1042-20261008-a01
test -d "$TASK_ROOT"
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then exit 1; fi
test ! -e "$TASK_ROOT/launch.log"
/mnt/local/conda-py311/envs/attention_bench/bin/python -u - "$TASK_ROOT" <<'PY'
import json,sys,time
from pathlib import Path
from scripts.stop_proxy_followup import ROOT,handed_off
root=Path(sys.argv[1]);deadline=time.monotonic()+1200
while time.monotonic()<deadline:
 if handed_off():break
 time.sleep(10)
else:raise RuntimeError('Old supervisor did not restore its verified burn')
supervisor=json.loads((ROOT/'supervised/supervisor.json').read_text())
burn=json.loads((ROOT/'supervised/burn-verified.json').read_text())
assert supervisor['training_status']=='failed' and supervisor.get('finished_at')
assert burn['collective_progress_verified'] and burn['session']=='proxy-followup-2500-20261008-a01-final-burn'
with (root/'stop.json').open('x') as handle:
 json.dump({'status':'old_queue_stopped_and_burn_verified','supervisor':supervisor,'burn':burn},handle,indent=2)
print('OLD_QUEUE_STOPPED_AND_BURN_VERIFIED',flush=True)
PY
printf -v TASK_CMD 'exec bash %q %q >%q 2>&1' \
  "$PWD/scripts/launch_proxy_10k_fresh.sh" "$TASK_ROOT" "$TASK_ROOT/launch.log"
tmux new-session -d -s "$TASK_SESSION" "$TASK_CMD"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 45
tail -n 80 "$TASK_ROOT/launch.log"
test "$(tmux display-message -p -t "$TASK_SESSION" '#{pane_dead}')" = 0
