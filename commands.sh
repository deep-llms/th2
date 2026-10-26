#1 +30+a
#th2-6y3o-efficient-kv-recovery-preflight-20261027-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/train/bin/python3.11 -u - <<'REMOTE'
import json, socket, hashlib, shutil
from pathlib import Path
from datetime import datetime,timezone
from scripts.gpu_process import inspect
from scripts.train_then_burn import BURN_HASH, GUARD_HASH, digest
assert socket.gethostname()=='thiennh-p6-6y3o-worker-0'
base=Path('/mnt/local/_outputs/deep-llms_th2')
previous=base/'6y3o-efficient-kv-mlp-20261027-a03'
last=base/'6y3o-efficient-kv-mlp-inference-20261027-a01'
print('PREFLIGHT_AT',datetime.now(timezone.utc).isoformat(),flush=True)
assert digest(Path('/mnt/local/_gpu_guard/gpu_guard.sh'))==GUARD_HASH
assert not Path('/mnt/local/_gpu_guard/DISABLED').exists()
assert digest(last/'source/resources/llm_pretrain_burn.py')==BURN_HASH
for root in [previous,last]:
 assert json.loads((root/'supervised/run/complete.json').read_text())['status']=='ok'
 assert json.loads((root/'supervised/burn-verified.json').read_text())['collective_progress_verified']
for arm in 'AB':
 p=previous/'supervised/run'/arm
 r=json.loads((p/'result.json').read_text())
 assert r['status']=='passed' and r['steps']==512 and not r['smoke']
 print('PARENT',arm,'result_sha256',digest(p/'result.json'),flush=True)
 for name in ['config.json','model.safetensors']:
  f=p/'final'/name
  print('PARENT_FILE',arm,name,f.stat().st_size,digest(f),flush=True)
print('DISK_FREE_GIB',shutil.disk_usage(base).free/1024**3,flush=True)
assert shutil.disk_usage(base).free>=200*1024**3
print('GPU_INSPECTION',json.dumps(inspect()),flush=True)
print('BURN_TAIL',(last/'supervised/burn.log').read_text()[-4000:],flush=True)
print('PREFLIGHT_OK',flush=True)
REMOTE
