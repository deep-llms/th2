#1 +30+a
#th2-6y3o-efficient-window-compact-health-20261027-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/train/bin/python3.11 -u - <<'REMOTE'
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib
from scripts.gpu_process import inspect
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-efficient-window-compact-20261027-a01')
print('HEALTH_AT',datetime.now(timezone.utc).isoformat(),flush=True)
for name in ['supervisor.log','accelerate-env.log','supervised/supervisor.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/run/run.json','supervised/run/window-compact-gate.log','supervised/run/window-profile.log','supervised/run/window-quality.log','supervised/run/complete.json','supervised/burn-verified.json']:
 p=root/name
 print('FILE',name,'EXISTS',p.is_file(),flush=True)
 if p.is_file():print(p.read_text()[-13000:],flush=True)
for name in ['supervised/run/gate.json','supervised/run/profile.json','supervised/run/quality.json']:
 p=root/name
 if p.is_file():
  obj=json.loads(p.read_text());print('RESULT',name,'SHA256',hashlib.sha256(p.read_bytes()).hexdigest(),flush=True)
  print(json.dumps(obj),flush=True)
p=root/'supervised/burn.log'
if p.is_file():print('BURN_TAIL',p.read_text()[-8000:],flush=True)
for name in ['inputs/manifest.json','source-manifest.json','accelerate-env.log','supervised/supervisor.json','supervised/burn-verified.json','supervised/run/complete.json','supervised/reclaim.json','supervised/gpus-free-before-training.json']:
 p=root/name
 if p.is_file():print('ARTIFACT_HASH',name,hashlib.sha256(p.read_bytes()).hexdigest(),flush=True)
print('GPU_INSPECTION',json.dumps(inspect()),flush=True)
p=Path('/mnt/local/_gpu_guard/DISABLED');print('GUARD_DISABLED',p.exists(),p.read_text() if p.exists() else '',flush=True)
print('HEALTH_COMPLETE',flush=True)
REMOTE
