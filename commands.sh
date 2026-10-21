#1 +30+a
#th2-6y3o-dot-diagnostics-gradient-launch-20261021-a02
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
TASK_PYTHON=/mnt/local/conda-py311/envs/train/bin/python3.11
TASK_ROOT=/mnt/local/_outputs/deep-llms_th2/6y3o-dot-diagnostics-gradient-20261021-a02
TASK_SESSION=dot_diagnostics_gradient_prepare_20261021_a02
test ! -e "$TASK_ROOT"
test ! -e "$TASK_ROOT.prepare.log"
test ! -e "$TASK_ROOT.prepare.py"
if tmux has-session -t "$TASK_SESSION" 2>/dev/null; then exit 1; fi
cat > "$TASK_ROOT.prepare.py" <<'PY'
from pathlib import Path
import json,os,socket,subprocess,sys
project=Path.cwd();sys.path.insert(0,str(project))
from scripts.native_baseline_pipeline import burn_gate
from scripts.gpu_process import inspect
from scripts.train_then_burn import burn_progress,digest,GUARD_HASH,BURN_HASH
from scripts.bdot_diagnostic_queue import write
out=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-diagnostics-gradient-20261021-a02')
base=out.parent
pred=base/'6y3o-dot-diagnostics-gradient-20261021-a01'
study=base/'6y3o-dot-diagnostics-20261021-a01'
assert socket.gethostname()=='thiennh-p6-6y3o-worker-0'
out.mkdir(parents=True,exist_ok=False)
try:
 burn_gate(pred)
 assert burn_progress((pred/'supervised/burn.log').read_text())
 assert json.loads((study/'supervised/run/complete.json').read_text())['status']=='ok'
 import shutil
 for name in ['original_results.tar.gz','original_results_manifest.json']:
  shutil.copy2(pred/name,out/name)
 assert digest(out/'original_results.tar.gz')==json.loads((out/'original_results_manifest.json').read_text())['sha256']
 (out/'previous_gradient').mkdir()
 for arm in ['P1-A','P1-B']:
  shutil.copy2(pred/'supervised/run'/arm/'result.json',out/'previous_gradient'/(arm+'.json'))
 shutil.copy2(pred/'supervised/run/run.json',out/'previous_gradient/run.json')
 subprocess.run([sys.executable,'scripts/bdot_diagnostic_queue.py','gradient-prepare',
   '--output',str(out/'prepared'),'--study',str(study)],check=True)
 burn_gate(pred)
 assert digest('/mnt/local/_gpu_guard/gpu_guard.sh')==GUARD_HASH
 assert digest(project/'resources/llm_pretrain_burn.py')==BURN_HASH
 live=inspect()
 assert live['host']=='thiennh-p6-6y3o-worker-0' and len(live['gpus'])==8
 assert all('B200' in g['name'] for g in live['gpus'])
 write(out/'preflight.json',live)
 subprocess.run([sys.executable,'scripts/bdot_b200_launch.py','--root',str(out),
    '--host',live['host'],'--session','dot_diagnostics_gradient_20261021_a02',
    '--jobs-config',str(out/'prepared/jobs.json')],check=True,
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'0,1,2,3,4,5,6,7'})
 print('DIAGNOSTICS_QUEUE_SUBMITTED',flush=True)
except BaseException as error:
 write(out/'preparation-failed.json',dict(status='failed',error=repr(error)))
 raise
PY
printf -v TASK_COMMAND 'exec env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false %q -u %q >%q 2>&1' "$TASK_PYTHON" "$TASK_ROOT.prepare.py" "$TASK_ROOT.prepare.log"
tmux new-session -d -s "$TASK_SESSION" "$TASK_COMMAND"
tmux set-option -w -t "$TASK_SESSION" remain-on-exit on
sleep 5
tail -20 "$TASK_ROOT.prepare.log"
echo DIAGNOSTICS_PREPARATION_SUBMITTED
