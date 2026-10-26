#1 +30+a
#th2-6y3o-kv-mlp-preflight-20261027-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/train/bin/python3.11 -u - <<'REMOTE'
from pathlib import Path
from datetime import datetime, timezone
import json, socket, importlib.metadata
from scripts.gpu_process import inspect
print('PREFLIGHT_AT',datetime.now(timezone.utc).isoformat(),flush=True)
print('HOST',socket.gethostname(),flush=True)
print('VERSIONS',json.dumps({n:importlib.metadata.version(n) for n in ['torch','transformers','accelerate','datasets']}),flush=True)
print('GPU_INSPECTION',json.dumps(inspect()),flush=True)
p=Path('/mnt/local/_gpu_guard/DISABLED')
print('GUARD_DISABLED',p.exists(),p.read_text() if p.exists() else '',flush=True)
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-efficient-window-compact-20261027-a01')
for name in ['supervised/supervisor.json','supervised/burn-verified.json']:
 p=root/name
 print('PREVIOUS',name,p.read_text() if p.exists() else 'MISSING',flush=True)
print('PREFLIGHT_COMPLETE',flush=True)
REMOTE
