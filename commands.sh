#1 +30+a
#th2-6y3o-efficient-kv-placement-final-20261027-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/train/bin/python3.11 -u - <<'READONLY'
import json,hashlib,socket
from pathlib import Path
from datetime import datetime,timezone
from scripts.gpu_process import inspect
assert socket.gethostname()=='thiennh-p6-6y3o-worker-0'
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-efficient-kv-placement-20261027-a01')
print('HEALTH_AT',datetime.now(timezone.utc).isoformat(),flush=True)
print('GPU_INSPECTION',json.dumps(inspect()),flush=True)
for rel in ['jobs.json','source-manifest.json','input-identity.json','accelerate-env.log','supervised/supervisor.json','supervised/burn-verified.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/run/complete.json','supervised/run/summary.json','supervised/run/gate-full50.json','supervised/run/gate-uniform25.json','supervised/run/gate-upper25.json','supervised/run/smoke/result.json','supervised/run/upper25/result.json','supervised/run/uniform25/result.json','supervised/run/full50/result.json','supervised/run/upper25/validation.json','supervised/run/uniform25/validation.json','supervised/run/full50/validation.json']:
 p=root/rel
 if p.is_file():
  raw=p.read_bytes()
  if len(raw)>2000000:raise ValueError('Unexpected oversized result '+rel)
  print('ARTIFACT',json.dumps(dict(path=rel,sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode())),flush=True)
 else:print('MISSING',rel,flush=True)
for p in sorted(root.rglob('*.log')):
 if p.stat().st_size:
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-6000));tail=f.read().decode(errors='replace')
  print('LOG_TAIL',json.dumps(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,text=tail)),flush=True)
print('HEALTH_COMPLETE',flush=True)
READONLY
