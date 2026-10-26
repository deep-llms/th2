#1 +30+a
#th2-6y3o-efficient-kv-mlp-inference-final-20261027-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/train/bin/python3.11 -u - <<'REMOTE'
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, shutil
from scripts.gpu_process import inspect
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-efficient-kv-mlp-inference-20261027-a01')
print('FINAL_AT',datetime.now(timezone.utc).isoformat(),flush=True)
for name in ['supervised/run/run.json','supervised/supervisor.json','supervised/burn-verified.json']:
 p=root/name
 print('FILE',name,flush=True)
 print(p.read_text() if p.is_file() else 'MISSING',flush=True)
for name in ['supervised/run/bf16-inference-quality.log','supervised/burn.log','supervisor.log']:
 p=root/name
 if p.is_file():print('LOG',name,p.read_text()[-6000:],flush=True)
p=root/'supervised/run/inference-quality.json'
if p.is_file():
 r=json.loads(p.read_text())
 print('INFERENCE_QUALITY',json.dumps({a:{k:v[k] for k in ['loss','ppl','versus_training_precision']} for a,v in r['arms'].items()}),flush=True)
print('GPU_INSPECTION',json.dumps(inspect()),flush=True)
guard=Path('/mnt/local/_gpu_guard/DISABLED')
print('GUARD_DISABLED',guard.exists(),guard.read_text() if guard.exists() else '',flush=True)
complete=root/'supervised/run/complete.json';burn=root/'supervised/burn-verified.json'
if complete.is_file() and burn.is_file() and not guard.exists():
 assert json.loads((root/'supervised/supervisor.json').read_text())['training_status']=='ok'
 assert json.loads(burn.read_text())['collective_progress_verified'] is True
 assert json.loads(complete.read_text())['status']=='ok'
 assert json.loads((root/'supervised/run/inference-quality.json').read_text())['status']=='passed'
 paths={'inference-quality.json':'supervised/run/inference-quality.json',
 'complete.json':'supervised/run/complete.json','supervisor.json':'supervised/supervisor.json',
 'burn-verified.json':'supervised/burn-verified.json','reclaim.json':'supervised/reclaim.json',
 'gpus-free.json':'supervised/gpus-free-before-training.json',
 'source-manifest.json':'source-manifest.json','accelerate-env.log':'accelerate-env.log'}
 for name in paths.values():assert (root/name).is_file() and (root/name).stat().st_size<25*1024**2,name
 target=root/'result_export';target.mkdir(exist_ok=False)
 receipt={}
 for name,source in paths.items():
  shutil.copyfile(root/source,target/name)
  receipt[name]=dict(source=source,sha256=hashlib.sha256((target/name).read_bytes()).hexdigest(),bytes=(target/name).stat().st_size)
 (target/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print('EXPORT_MANIFEST',json.dumps(receipt),flush=True)
 print('EXPORT_MANIFEST_SHA256',hashlib.sha256((target/'manifest.json').read_bytes()).hexdigest(),flush=True)
 print('RESULT_EXPORT_READY',flush=True)
else:print('NOT_FINISHED',flush=True)
REMOTE
