#1 +30+a
#th2-6y3o-efficient-router-pilot-health-20261026-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/train/bin/python3.11 -u - <<'REMOTE'
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib
from scripts.gpu_process import inspect
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-efficient-router-pilot-20261026-a01')
print('HEALTH_AT',datetime.now(timezone.utc).isoformat(),flush=True)
for name in ['supervisor.log','supervised/supervisor.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/run/run.json','supervised/run/router-gpu-correctness.log','supervised/run/gpu-check.json','supervised/run/matched-router-pilot.log','supervised/run/cached-decode-profile.log','supervised/run/complete.json','supervised/burn-verified.json']:
 p=root/name
 print('FILE',name,'EXISTS',p.is_file(),flush=True)
 if p.is_file(): print(p.read_text()[-22000:],flush=True)
for name in ['supervised/run/pilot/training.json','supervised/run/pilot/evaluation.partial.json','supervised/run/pilot/evaluation.json','supervised/run/decode-profile.json']:
 p=root/name
 if p.is_file():
  obj=json.loads(p.read_text())
  print('RESULT',name,'SHA256',hashlib.sha256(p.read_bytes()).hexdigest(),flush=True)
  if 'methods' in obj:
   obj={k:v for k,v in obj.items() if k!='regret'}
   obj['methods']={k:{kk:vv for kk,vv in v.items() if kk!='per_document_nll'} for k,v in obj['methods'].items()}
  if 'trainer_args' in obj: obj.pop('trainer_args')
  print(json.dumps(obj),flush=True)
print('GPU_INSPECTION',json.dumps(inspect()),flush=True)
p=Path('/mnt/local/_gpu_guard/DISABLED')
print('GUARD_DISABLED',p.exists(),p.read_text() if p.exists() else '',flush=True)
print('HEALTH_COMPLETE',flush=True)
REMOTE
