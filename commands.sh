#1 +30+a
#th2-q359-template-realdata-final-20261017-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/eval_fa4/bin/python -B - <<'PY'
import json,shutil
from pathlib import Path
from datetime import datetime,timezone
from scripts.verified_gpu_reclaim import inspect
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-template-realdata-20261017-a02')
print('TIME',datetime.now(timezone.utc).isoformat())
print('INSPECTION',json.dumps(inspect()))
print('DISK',shutil.disk_usage(root)._asdict())
for p in [root/'supervised/supervisor.json',root/'supervised/run/complete.json',root/'supervised/burn-verified.json']:
 if p.exists():print('RESULT',json.dumps({'path':str(p),'data':json.loads(p.read_text())}))
for backend in ('sdpa','fa4'):
 p=root/'supervised/run'/backend/'training/training_recipe.json'
 if p.exists():
  r=json.loads(p.read_text());print('RECIPE',json.dumps({'backend':backend,'execution':r['execution'],'warmup_steps':r['training']['warmup_steps'],'max_steps':r['training']['max_steps']}))
p=root/'supervised/burn.log'
if p.exists():print('BURN_LOG','\n'.join(p.read_text().splitlines()[-10:]))
PY
