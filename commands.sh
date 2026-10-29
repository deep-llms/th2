#1 +30+a
#th2-6y3o-a2-stage1b-fwe-inventory-20261030-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PY'
import hashlib,json,socket,shutil
from pathlib import Path
from datetime import datetime,timezone
from scripts.gpu_process import inspect
from scripts.a2_preflight_inputs import _parent_asset_plan
from scripts.a2_prepare_inputs import verify_inputs
assert socket.gethostname()=='thiennh-p6-6y3o-worker-0'
print('AT',datetime.now(timezone.utc).isoformat(),flush=True)
print('GPU_INSPECTION',json.dumps(inspect()),flush=True)
print('DISK',json.dumps(dict(zip(('total','used','free'),shutil.disk_usage('/mnt/local')))),flush=True)
plan=json.loads(Path('resources/a2_inputs_20261028.json').read_text())
for parent in ('P300','P1T','S72'):
 verify_inputs(_parent_asset_plan(plan,parent,'fwe'))
 print('MODEL_VERIFIED',parent,json.dumps(plan['assets']['fwe/'+parent]),flush=True)
root=Path('/mnt/local/_data/deep-llms_th2/a2_corpora_30b_20261029/packed/fineweb_edu')
p=root/'manifest.json';h=hashlib.sha256(p.read_bytes()).hexdigest();assert h=='0685a790f3c60bf0c02f2e0679910ada71d9c3fefd95a99c53578b201b87f3dd'
m=json.loads(p.read_text());sizes={}
for n,s in m['streams'].items():
 for suffix,mult in [('bin',4),('starts',1)]:
  p=root/(n+'.'+suffix);assert p.stat().st_size==s['tokens']*mult;sizes[p.name]=p.stat().st_size
print('FWE_MANIFEST_VERIFIED',h,json.dumps(sizes),flush=True)
for rel in ['supervised/supervisor.json','supervised/burn-verified.json']:
 p=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-a2-stage1-downstream-20261029-a01')/rel
 print('PREDECESSOR',rel,p.read_text(),flush=True)
print('STAGE1B_INVENTORY_COMPLETE',flush=True)
PY
