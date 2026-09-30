#1 +60+a
#th2-78gg-AB-seed123-status-20260930-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-78gg-worker-0
export CUDA_VISIBLE_DEVICES=''
/mnt/local/conda-py311/envs/train_env/bin/python -u - <<'PY'
import json, hashlib, subprocess
from datetime import datetime, timezone
from pathlib import Path
from scripts.verified_gpu_reclaim import inspect
root = Path('/mnt/local/_outputs/deep-llms_th2/deep-kv-AB-seed123-2500-20260930-a01')
print('STATUS_AT', datetime.now(timezone.utc).isoformat(), flush=True)
for name in ['recipe.json','jobs.json','seed_provenance.json','preflight.json','gpu_inspection.json','production/supervisor.json','production/reclaim.json',
             'production/gpus-free-before-training.json','production/run/run.json',
             'production/run/smoke/smoke_complete.json','production/run/complete.json',
             'production/burn-verified.json']:
 path=root/name
 if path.is_file():
  raw=path.read_bytes()
  print('ARTIFACT_JSON',json.dumps(dict(path=name,sha256=hashlib.sha256(raw).hexdigest(),content=raw.decode())),flush=True)
for folder in [root/'production/run',root/'production/run/smoke/run']:
 for arm in ['A','B']:
  path=folder/arm
  for name in ['result.json','train_config.json']:
   if (path/name).is_file():
    raw=(path/name).read_bytes()
    print('ARTIFACT_JSON',json.dumps(dict(path=str((path/name).relative_to(root)),sha256=hashlib.sha256(raw).hexdigest(),content=raw.decode())),flush=True)
  states=sorted(path.glob('checkpoint-*/trainer_state.json'),key=lambda p:int(p.parent.name.split('-')[-1]))
  if states:
   state=json.loads(states[-1].read_text())
   print('CHECKPOINT',str(path.relative_to(root)),state['global_step'],json.dumps(state['log_history'][-3:]),flush=True)
  for p in sorted(path.glob('smoke-step*-rank0.json')):
   print('SMOKE_TELEMETRY',str(p.relative_to(root)),p.read_text(),flush=True)
def tail(path):
 if path.is_file():
  with path.open('rb') as f:
   f.seek(max(0,path.stat().st_size-16000))
   lines=f.read().decode(errors='replace').replace('\r','\n').splitlines()
  print('LOG_TAIL',str(path.relative_to(root)),flush=True)
  print('\n'.join([line for line in lines if line.strip()][-14:]),flush=True)
tail(root/'launch.log')
for folder in [root/'production/run',root/'production/run/smoke',root/'production/run/smoke/run']:
 for path in sorted(folder.glob('*.log')):tail(path)
print('LIVE_GPU_INSPECTION',json.dumps(inspect()),flush=True)
print('TMUX_PANE',subprocess.run(['tmux','display-message','-p','-t','deep-kv-AB-seed123-2500-20260930-a01','#{pane_dead}'],capture_output=True,text=True).stdout.strip(),flush=True)
PY
