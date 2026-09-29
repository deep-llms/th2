#1 +60+a
#th2-78gg-capacity-status-20260929-a04
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-78gg-worker-0
export CUDA_VISIBLE_DEVICES=''
/mnt/local/conda-py311/envs/train_env/bin/python - <<'PY'
import json
from pathlib import Path
from datetime import datetime,timezone
from scripts.verified_gpu_reclaim import inspect
root=Path('/mnt/local/_outputs/deep-llms_th2/capacity-20260929-a01')
print('CAPACITY_STATUS',datetime.now(timezone.utc).isoformat(),flush=True)
print('INSPECTION',json.dumps(inspect()),flush=True)
for path in [root/'launch.log',root/'production/run/capacity.log']:
 print('LOG_TAIL',str(path),flush=True)
 if path.exists():print('\n'.join(path.read_text().splitlines()[-20:]),flush=True)
for relative in ['production/run/run.json','production/supervisor.json','production/run/probes/progress.json','production/run/probes/capacity-summary.json']:
 path=root/relative
 if path.exists():
  value=json.loads(path.read_text())
  if 'results' in value:
   value['results']={n:{k:v for k,v in r.items() if k not in ('input_rows','validations','free_after')} for n,r in value['results'].items()}
  print('ARTIFACT',relative,json.dumps(value),flush=True)
paths=sorted((root/'production/run/probes').glob('*.log'),key=lambda p:p.stat().st_mtime)
if paths:
 print('LATEST_ATTEMPT_LOG',paths[-1].name,flush=True)
 print('\n'.join(paths[-1].read_text().splitlines()[-30:]),flush=True)
PY
