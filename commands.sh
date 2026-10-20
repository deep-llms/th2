#1 +30+a
#th2-6y3o-dot-N-status-20261020-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
python3 - <<'PY'
from pathlib import Path
import datetime,json
from scripts.gpu_process import inspect
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-N-20261020-a01')
print('OBSERVED_UTC',datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
for name in ['preparation-failed.json','control-comparison.json','cache-ready.json','launch.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/run/run.json','supervised/supervisor.json','supervised/burn-verified.json']:
 p=root/name
 if not p.exists():continue
 j=json.loads(p.read_text())
 if name=='cache-ready.json':j={k:j[k] for k in ['status','seconds','train_sequences','validation_sequences']}
 print('RECEIPT',name,json.dumps(j),flush=True)
for p in [Path(str(root)+'.prepare.log'),root/'supervisor.log',root/'supervised/run/N-A.log',root/'accelerate_env.txt']:
 if not p.exists():continue
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-2200));s=f.read().decode(errors='replace')
 print('LOG_TAIL',str(p),'\n'+s,flush=True)
path=root/'prepared/runs/N-A'
if path.exists():
 print('CHECKPOINTS',sorted(p.name for p in path.glob('checkpoint-*')),flush=True)
 for name in ['metrics.jsonl','train_results.json','final_eval.json']:
  p=path/name
  if not p.exists():continue
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-4500));s=f.read().decode(errors='replace')
  print('ARM_RESULT',name,'\n'+s,flush=True)
print('GPU_LIVE',json.dumps(inspect()),flush=True)
PY
