#1 +30+a
#th2-6y3o-dot-R-status-20261021-a02
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
python3 - <<'REMOTE_PY'
from pathlib import Path
import datetime,json,hashlib
from zoneinfo import ZoneInfo
from scripts.gpu_process import inspect
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-R-20261021-a01')
print('OBSERVED_SGT',datetime.datetime.now(ZoneInfo('Asia/Singapore')).isoformat(),flush=True)
for name in ['preparation-failed.json','supervised/run/run.json','supervised/run/complete.json','supervised/run/R.verified.json','supervised/supervisor.json','supervised/burn-verified.json']:
 p=root/name
 if p.exists():print('RECEIPT',name,p.read_text().replace('\n',' '),flush=True)
path=root/'prepared/runs/R'
for name in ['metrics.jsonl','train_results.json','final_eval.json']:
 p=path/name
 if not p.exists():continue
 if name=='metrics.jsonl':
  lines=p.read_text().splitlines()
  train=[json.loads(l) for l in lines if '"grad_norm"' in l]
  val=[json.loads(l) for l in lines if '"eval_loss"' in l]
  print('RECENT_TRAIN',json.dumps(train[-3:]),flush=True)
  print('RECENT_EVAL',json.dumps(val[-3:]),flush=True)
 else:print('ARM_RESULT',name,p.read_text().replace('\n',' '),flush=True)
 print('FILE_HASH',name,hashlib.sha256(p.read_bytes()).hexdigest(),flush=True)
checkpoints=sorted(path.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
print('CHECKPOINTS',json.dumps([p.name for p in checkpoints]),flush=True)
final=path/'checkpoint-4768'
if final.is_dir():
 names=['trainer_state.json','optimizer.pt','scheduler.pt']+[f'rng_state_{i}.pth' for i in range(8)]
 missing=[n for n in names if not (final/n).is_file() or (final/n).stat().st_size==0]
 state=json.loads((final/'trainer_state.json').read_text())
 weights=[p for p in final.glob('*.safetensors') if p.stat().st_size]
 print('CHECKPOINT_CHECK',json.dumps(dict(step=state['global_step'],count=len(checkpoints),missing=missing,weight_files=len(weights))),flush=True)
for name in ['supervisor.log','supervised/burn.log','supervised/run/R.log']:
 p=root/name
 if p.exists():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-2500));tail=f.read().decode(errors='replace')
  print('LOG_TAIL',name,'\n'+tail,flush=True)
live=inspect();print('GPU_LIVE',json.dumps(live),flush=True)
if (root/'supervised/burn-verified.json').exists():
 from scripts.native_baseline_pipeline import burn_gate
 burn_gate(root)
 print('CURRENT_BURN_HANDOFF_VERIFIED',flush=True)
REMOTE_PY
