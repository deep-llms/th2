#1 +60+a
#th2-tjx3-p7-checks-monitor-20261006-a02
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
date -u
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'PY'
import json
from pathlib import Path
from scripts.gpu_status import snapshot
root=Path('/mnt/local/_outputs/deep-llms_th2/p7-checks-20261006-a01');run=root/'supervised/run'
for name in ['launch.log','preflight.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/supervisor.json','supervised/run/run.json','supervised/run/complete.json','supervised/burn-verified.json']:
 p=root/name
 if p.is_file():print('FILE',name,p.read_text()[-10000:],flush=True)
for p in sorted(run.glob('*.log')):
 print('JOB_LOG',p.name,p.read_text()[-7000:],flush=True)
for p in sorted(run.glob('*.json')):
 if p.name.startswith(('numerics','cross-backend')):print('NUMERICS',p.name,p.read_text(),flush=True)
for backend in ('fa4','sdpa'):
 for p in sorted((run/backend).glob('*.json')):
  print('STAGE',backend,p.name,p.read_text()[-15000:],flush=True)
 for folder in sorted((run/backend/'seed-42').glob('*')):
  p=folder/'result.json'
  if p.is_file():
   r=json.loads(p.read_text());print('RESULT',backend,folder.name,json.dumps({k:r[k] for k in ('status','global_step','training_cost','evaluation')}),flush=True)
print('GPUS',json.dumps(snapshot(list(range(8)))),flush=True)
PY
