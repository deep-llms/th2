#1 +60+a
#th2-78gg-FG-readonly-status-20260928-2303
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/mnt/local/conda-py311/envs/train_env/bin/python - <<'CHECK'
import json, socket, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
from scripts.gpu_status import snapshot
root=Path('/mnt/local/_outputs/deep-llms_th2/deep-kv-FG-2500-20260928-a01')
print('FG_STATUS',datetime.now(timezone.utc).isoformat(),socket.gethostname(),flush=True)
for name in ['production/supervisor.json','production/run/run.json','production/run/complete.json','production/run/comparison.json','production/burn-verified.json']:
 p=root/name
 print('ARTIFACT',name,p.read_text() if p.exists() else 'not yet present',flush=True)
for arm in 'FG':
 d=root/'production/run'/arm
 for name in ['result.json','trainer_state.json']:
  p=d/name
  if p.exists():
   data=json.loads(p.read_text())
   if name=='trainer_state.json':
    data={k:data.get(k) for k in ['global_step','max_steps','epoch','log_history']}
    data['log_history']=data['log_history'][-3:]
   print('ARM_ARTIFACT',arm,name,json.dumps(data),flush=True)
 print('CHECKPOINTS',arm,sorted(p.name for p in d.glob('checkpoint-*')),flush=True)
 p=root/'production/run'/f'arm-{arm}.log'
 if p.exists():
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-7000))
   text=f.read().decode(errors='replace')
  print('LOG_TAIL',arm,text,flush=True)
print('GPUS',json.dumps(snapshot(list(range(8)))),flush=True)
p=root/'production/burn.log'
if p.exists():
 for sample in range(2):
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-5000))
   print('BURN_PROGRESS',sample,f.read().decode(errors='replace'),flush=True)
  if sample==0: time.sleep(12)
print('FG_STATUS_COMPLETE',datetime.now(timezone.utc).isoformat(),flush=True)
CHECK
