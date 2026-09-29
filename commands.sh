#1 +60+a
#th2-78gg-BFG-10000-retry-audit-20260929-a04
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-78gg-worker-0
export CUDA_VISIBLE_DEVICES=''
/mnt/local/conda-py311/envs/train_env/bin/python - <<'PY'
import hashlib,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
from scripts.verified_gpu_reclaim import inspect
root=Path('/mnt/local/_outputs/deep-llms_th2/deep-kv-BFG-10000-20260929-a02')
print('LIVE_AUDIT',datetime.now(timezone.utc).isoformat(),flush=True)
paths=[root/p for p in ['preflight.json','previous-handoff.json','optimized-recipe.json','jobs.json','gpu_inspection.json',
 'production/reclaim.json','production/gpus-free-before-training.json','production/supervisor.json',
 'production/run/run.json','production/run/resume-smoke/verified.json',
 'production/run/continuation/resume_inputs.json','production/run/continuation/comparison.json',
 'production/burn-verified.json','production/handoff-error.json']]
for mode in ('control','optimized'):
 paths.append(root/f'production/run/resume-smoke/{mode}/resume_inputs.json')
 for arm in 'BFG':
  folder=root/f'production/run/resume-smoke/{mode}/{arm}'
  paths.extend(folder.glob('resume-check-*.json'))
  paths.extend(folder.glob('resume-transition-*.json'))
  paths.extend(folder/p for p in ['train_config.json','result.json'])
for arm in 'BFG':
 folder=root/f'production/run/continuation/{arm}'
 paths.extend(folder.glob('resume-transition-*.json'))
 paths.extend(folder/p for p in ['train_config.json','result.json'])
for path in paths:
 if path.is_file():
  raw=path.read_bytes()
  print('ARTIFACT_JSON',json.dumps(dict(path=str(path.relative_to(root)),sha256=hashlib.sha256(raw).hexdigest(),content=raw.decode())),flush=True)
for name in ['launch.log','production/run/verify-resume.log','production/run/smoke-optimized-B.log','production/run/smoke-optimized-F.log','production/run/smoke-optimized-G.log','production/run/arm-B.log','production/run/arm-F.log','production/run/arm-G.log']:
 path=root/name
 if path.exists():
  with path.open('rb') as f:
   f.seek(max(0,path.stat().st_size-18000))
   lines=f.read().decode(errors='replace').replace('\r','\n').splitlines()
  print('LOG_TAIL',name,flush=True)
  print('\n'.join([s for s in lines if s.strip()][-40:]),flush=True)
print('LIVE_GPU_INSPECTION',json.dumps(inspect()),flush=True)
print('TMUX_PANE',subprocess.run(['tmux','display-message','-p','-t','deep-kv-BFG-10000-20260929-a02','#{pane_dead}'],capture_output=True,text=True).stdout.strip(),flush=True)
PY
