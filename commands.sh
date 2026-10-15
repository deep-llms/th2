#1 +60+a
#th2-q359-p7-exploration8-20261015-a03
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
date -u
printf 'SOURCE_IMPLEMENTATION 6ce50b3\n'
nvidia-smi
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'HANDOFF'
from pathlib import Path
import json,time
from scripts.train_then_burn import burn_progress,approved_launcher,process,BURN_HASH
from scripts.gpu_status import snapshot
from scripts.verified_gpu_reclaim import ownership
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-p7-exploration8-20261015-a02/supervised')
state,run,burn=(json.loads((root/name).read_text()) for name in ('supervisor.json','run/run.json','burn-verified.json'))
assert state.get('finished_at') and state['training_status']=='failed'
assert run['status']=='failed' and run['error']=='numerics exited with code 1'
assert len(run['jobs'])==1 and run['jobs'][0]['name']=='numerics'
assert burn['collective_progress_verified'] and state['burn']['collective_progress_verified']
assert burn['session']==state['burn']['session']
assert not Path('/mnt/local/_gpu_guard/DISABLED').exists()
status=snapshot(list(range(8)));workers={p for g in status for p in g['pids']}
assert ownership(status)==ownership(burn['gpus']) and len(workers)==8
source=str(Path.cwd()/'resources/llm_pretrain_burn.py')
assert all(approved_launcher(process(pid)['ppid'],{source:BURN_HASH}) for pid in workers)
log=root/'burn.log'
assert burn_progress(log.read_text()) and time.time()-log.stat().st_mtime<120
print('FAILED_GATE_BURNS_RESTORED_AND_LIVE',json.dumps(burn),flush=True)
HANDOFF
TASK_ROOT=/mnt/local/_outputs/deep-llms_th2/q359-p7-exploration8-20261015-a03
TASK_SESSION=q359-p7-exploration8-20261015-a03
TASK_LOG=/mnt/local/_outputs/deep-llms_th2/q359-p7-exploration8-20261015-a03.launch.log
TASK_PREDECESSOR=/mnt/local/_outputs/deep-llms_th2/q359-p7-target-integrated-20261015-a01
TASK_REFERENCE=/mnt/local/_outputs/deep-llms_th2/q359-proxy12-eval-remaining27-20261014-a01
TASK_ROWS=/mnt/local/_outputs/deep-llms_th2/q359-p7-teacher-probe-20261015-a01/rows.json
test ! -e "$TASK_ROOT"
test ! -e "$TASK_LOG"
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then
  echo 'REFUSE: exploration session already exists' >&2
  exit 1
fi
printf -v TASK_COMMAND 'bash scripts/launch_p7_exploration.sh %q %q %q %q %q >%q 2>&1' \
  "$TASK_ROOT" thiennh-p6-q359-worker-0 "$TASK_PREDECESSOR" "$TASK_REFERENCE" "$TASK_ROWS" "$TASK_LOG"
tmux new-session -d -s "$TASK_SESSION" "$TASK_COMMAND"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 5
tmux list-panes -t "$TASK_SESSION" -F '#{pane_dead} #{pane_pid} #{pane_current_command}'
tail -n 25 "$TASK_LOG"
