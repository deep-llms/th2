#1 +60+a
#th2-q359-expanded-paired-20k-20261012-a03
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
/mnt/local/conda-py311/envs/eval_fa4/bin/python3.11 -u - <<'PYVERIFY'
import json,time,subprocess
from pathlib import Path
from eval.suite import collect,selected_protocols
old=Path('/mnt/local/_outputs/deep-llms_th2/q359-expanded-paired-20k-20261012-a02')
for _ in range(60):
    receipt=json.loads((old/'supervised/supervisor.json').read_text())
    dead=subprocess.check_output(['tmux','display-message','-p','-t',old.name,'#{pane_dead}'],text=True).strip()
    if receipt.get('finished_at') and dead=='1':break
    time.sleep(5)
else:raise RuntimeError('Previous supervisor has not finished handoff')
assert receipt['training_status']=='failed' and receipt['burn']['collective_progress_verified'] and not receipt.get('handoff_error'),receipt
print('PREVIOUS_FAILURE_BURN_RESTORATION_VERIFIED',receipt['finished_at'],flush=True)
ck=json.loads((old/'preflight.json').read_text())['checkpoints']
report=collect(old/'supervised/run/expanded-smoke',[ck['A20'],ck['P20']],2,selected_protocols(['zero','five','math','continuation'],True))
with (old/'smoke-collection-reviewed.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
print('ALL_EIGHT_REAL_SMOKE_OUTPUTS_VALIDATED',report['status'],flush=True)
PYVERIFY
TASK_ROOT=/mnt/local/_outputs/deep-llms_th2/q359-expanded-paired-20k-20261012-a03
TASK_SESSION=q359-expanded-paired-20k-20261012-a03
test ! -e "$TASK_ROOT"
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then exit 1; fi
mkdir -p "$TASK_ROOT"
tmux new-session -d -s "$TASK_SESSION" "exec bash scripts/launch_expanded_evaluation.sh '$TASK_ROOT' thiennh-p6-q359-worker-0 2000 resources/position_extension_selection_20261012.json >'$TASK_ROOT/supervisor.log' 2>&1"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 45
tail -35 "$TASK_ROOT/supervisor.log"
if [ "$(tmux display-message -p -t "$TASK_SESSION" '#{pane_dead}')" = 1 ]; then
  /mnt/local/conda-py311/envs/eval_fa4/bin/python3.11 - "$TASK_ROOT" <<'PYREMOTE'
import json,sys
from pathlib import Path
r=json.loads((Path(sys.argv[1])/'supervised/supervisor.json').read_text())
assert r['training_status']=='ok' and r['training_returncode']==0 and r['burn']['collective_progress_verified'],r
print('EXPANDED_PAIRED_STUDY_AND_BURN_COMPLETE',flush=True)
PYREMOTE
else
  echo 'EXPANDED_PAIRED_SUPERVISOR_ACTIVE: downloads/preflight or evaluation queue.'
fi
