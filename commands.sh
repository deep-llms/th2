#1 +60+a
#th2-78gg-BFG-status-20260929-0532
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-78gg-worker-0
export CUDA_VISIBLE_DEVICES=''
/mnt/local/conda-py311/envs/train_env/bin/python - <<'PY'
import hashlib, json, socket, time
from datetime import datetime, timezone
from pathlib import Path
from scripts.verified_gpu_reclaim import inspect
root=Path('/mnt/local/_outputs/deep-llms_th2/deep-kv-BFG-5000-20260929-a01')
print('STATUS_CHECK',datetime.now(timezone.utc).isoformat(),socket.gethostname(),flush=True)
for name in ['production/supervisor.json','production/run/run.json','production/run/complete.json','production/run/comparison.json','production/burn-verified.json']:
 path=root/name
 if path.exists():
  raw=path.read_bytes()
  print('ARTIFACT',name,'SHA256',hashlib.sha256(raw).hexdigest(),flush=True)
  print(raw.decode(),flush=True)
 else:print('ABSENT',name,flush=True)
for arm in 'BFG':
 path=root/'production/run'/arm/'result.json'
 if path.exists():
  raw=path.read_bytes()
  print('ARTIFACT',str(path.relative_to(root)),'SHA256',hashlib.sha256(raw).hexdigest(),flush=True)
  print(raw.decode(),flush=True)
 state=root/'production/run'/arm/'trainer_state.json'
 if state.exists():
  value=json.loads(state.read_text())
  print('TRAINER_STATE',arm,json.dumps({k:value[k] for k in ['global_step','max_steps','epoch']}),flush=True)
 path=root/'production/run'/f'arm-{arm}.log'
 if path.exists():
  with path.open('rb') as f:
   f.seek(max(0,path.stat().st_size-18000))
   lines=f.read().decode(errors='replace').replace('\r','\n').splitlines()
  print('LOG_TAIL',arm,flush=True)
  print('\n'.join([line for line in lines if line.strip()][-24:]),flush=True)
print('LIVE_GPU_INSPECTION',json.dumps(inspect()),flush=True)
if (root/'production/burn-verified.json').exists():
 for sample in range(2):
  path=root/'production/burn.log'
  with path.open('rb') as f:
   f.seek(max(0,path.stat().st_size-12000))
   lines=f.read().decode(errors='replace').splitlines()
  print('BURN_SAMPLE',sample,datetime.now(timezone.utc).isoformat(),flush=True)
  print('\n'.join(lines[-5:]),flush=True)
  if sample==0:time.sleep(12)
PY
