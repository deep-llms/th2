#1 +30+a
#th2-6y3o-efficient-kv-mlp-health-20261027-a07
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/train/bin/python3.11 -u - <<'REMOTE'
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, subprocess
from scripts.gpu_process import inspect
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-efficient-kv-mlp-20261027-a03')
print('HEALTH_AT',datetime.now(timezone.utc).isoformat(),flush=True)
for name in ['supervisor.log','accelerate-env.log','supervised/supervisor.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/run/run.json','supervised/run/kv-mlp-gpu-gate.log','supervised/run/kv-mlp-calibration.log','supervised/run/kv-mlp-smoke.log','supervised/run/complete.json','supervised/burn-verified.json'] + ['supervised/run/kv-mlp-'+a+'.log' for a in 'ABCDEF']:
 p=root/name
 print('FILE',name,'EXISTS',p.is_file(),flush=True)
 if p.is_file():print(p.read_text()[-9500:],flush=True)
for name in ['probe/manifest.json','supervised/run/gpu-check.json','supervised/run/plan.json']:
 p=root/name
 if p.is_file():
  obj=json.loads(p.read_text());print('RESULT',name,'SHA256',hashlib.sha256(p.read_bytes()).hexdigest(),flush=True)
  print(json.dumps({k:v for k,v in obj.items() if k not in ['documents','verified_shard_files','ranks_dense','ranks_shared','arms']}),flush=True)
p=root/'supervised/burn.log'
if p.is_file():print('BURN_TAIL',p.read_text()[-4000:],flush=True)
print('GPU_INSPECTION',json.dumps(inspect()),flush=True)
p=Path('/mnt/local/_gpu_guard/DISABLED');print('GUARD_DISABLED',p.exists(),p.read_text() if p.exists() else '',flush=True)
for line in subprocess.check_output(['ps','-eo','pid,ppid,etime,pcpu,args'],text=True).splitlines():
 if ('efficient_llm.diagnostic prepare' in line or 'scripts.launch_group_diagnostic' in line) and not 'bash -c' in line:print('PREP_PROCESS',line,flush=True)
print('HEALTH_COMPLETE',flush=True)
REMOTE
