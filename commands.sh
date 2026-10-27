#1 +30+a
#th2-6y3o-a2-inputs-health-20261028-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PY'
import json,hashlib,socket,subprocess
from pathlib import Path
from datetime import datetime,timezone
from scripts.gpu_process import inspect
assert socket.gethostname()=='thiennh-p6-6y3o-worker-0'
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-a2-inputs-20261028-a01')
print('HEALTH_AT',datetime.now(timezone.utc).isoformat(),flush=True)
print('GPU_INSPECTION',json.dumps(inspect()),flush=True)
for rel in ['source-manifest.json','jobs.json','run/run.json','run/complete.json','run/inputs/verified_inputs.json','run/inputs/complete.json','run/inputs/failure.json','run/inputs/data/progress.json','run/inputs/data/manifest.json','run/inputs/secondary_map.json',*[f'run/inputs/secondary/{ds}/manifest.json' for ds in ('dclm','fwe','nem')]]:
 p=root/rel
 if p.is_file():
  raw=p.read_bytes()
  if len(raw)>2500000:raise ValueError('Oversized small artifact')
  print('ARTIFACT',json.dumps(dict(path=rel,sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode())),flush=True)
 else:print('MISSING',rel,flush=True)
for rel in ('runner.log','run/prepare.log'):
 p=root/rel
 if p.is_file():
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-7000));tail=f.read().decode(errors='replace')
  print('LOG_TAIL',json.dumps(dict(path=rel,bytes=p.stat().st_size,text=tail)),flush=True)
plan=json.loads(Path('resources/a2_inputs_20261028.json').read_text())
print('ASSET_FILES',json.dumps([dict(path=i['path'],bytes=Path(i['path']).stat().st_size if Path(i['path']).is_file() else None) for i in plan['downloads']]),flush=True)
print('SESSION',subprocess.run(['tmux','display-message','-p','-t','a2_inputs_20261028_a01','#{pane_dead} #{pane_pid}'],capture_output=True,text=True).stdout.strip(),flush=True)
print('HEALTH_COMPLETE',flush=True)
PY
