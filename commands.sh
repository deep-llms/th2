#1 +30+a
#th2-6y3o-a2-review-health-20261028-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'READONLY'
import json,hashlib,socket
from pathlib import Path
from datetime import datetime,timezone
from scripts.gpu_process import inspect
assert socket.gethostname()=='thiennh-p6-6y3o-worker-0'
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-a2-review-20261028-a01')
print('HEALTH_AT',datetime.now(timezone.utc).isoformat(),flush=True)
print('GPU_INSPECTION',json.dumps(inspect()),flush=True)
for rel in ['jobs.json', 'source-manifest.json', 'inspection.json', 'accelerate-env.log', 'supervised/supervisor.json', 'supervised/burn-verified.json', 'supervised/reclaim.json', 'supervised/gpus-free-before-training.json', 'supervised/run/complete.json', 'supervised/run/run.json', 'supervised/run/probe/reference.json', 'supervised/run/probe/ddp_result.json', 'supervised/run/probe/roundtrip-w1.json', 'supervised/run/probe/roundtrip-w8.json']:
 p=root/rel
 if p.is_file():
  raw=p.read_bytes()
  if len(raw)>2000000:raise ValueError('Unexpected oversized artifact '+rel)
  print('ARTIFACT',json.dumps(dict(path=rel,sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode())),flush=True)
 else:print('MISSING',rel,flush=True)
for p in sorted(root.rglob('*.log')):
 if p.is_file() and p.stat().st_size:
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-2500));tail=f.read().decode(errors='replace')
  print('LOG_TAIL',json.dumps(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,text=tail)),flush=True)
print('HEALTH_COMPLETE',flush=True)
READONLY
