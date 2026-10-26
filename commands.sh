#1 +30+a
#th2-6y3o-efficient-joint-groups-health-20261026-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/train/bin/python3.11 -u - <<'REMOTE'
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib
from scripts.gpu_process import inspect
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-efficient-joint-groups-20261026-a01')
print('HEALTH_AT',datetime.now(timezone.utc).isoformat(),flush=True)
for name in ['supervisor.log','supervised/supervisor.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/run/run.json','supervised/run/gpu-correctness.log','supervised/run/gpu-check.json','supervised/run/joint-groups.log','supervised/run/complete.json','supervised/burn-verified.json']:
 p=root/name
 print('FILE',name,'EXISTS',p.is_file(),flush=True)
 if p.is_file():
  print(p.read_text()[-35000:],flush=True)
for name in ['diagnostic.json','diagnostic.partial.json']:
 p=root/'supervised/run'/name
 if p.is_file():
  obj=json.loads(p.read_text())
  print('RESULT',name,'SHA256',hashlib.sha256(p.read_bytes()).hexdigest(),flush=True)
  print(json.dumps({'status':obj['status'],'budgets':{k:{m:{kk:vv for kk,vv in v.items() if kk!='per_document_nll'} for m,v in b['methods'].items()} for k,b in obj['budgets'].items()}}),flush=True)
print('GPU_INSPECTION',json.dumps(inspect()),flush=True)
p=Path('/mnt/local/_gpu_guard/DISABLED')
print('GUARD_DISABLED',p.exists(),p.read_text() if p.exists() else '',flush=True)
print('HEALTH_COMPLETE',flush=True)
REMOTE
