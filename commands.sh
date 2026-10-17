#1 +30+a
#th2-q359-template-realdata-review-20261017-a05
set -euo pipefail
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/eval_fa4/bin/python -B - <<'PY'
import json
from pathlib import Path
from datetime import datetime,timezone
from scripts.verified_gpu_reclaim import inspect
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-template-realdata-20261017-a02')
print('TIME',datetime.now(timezone.utc).isoformat())
print('INSPECTION',json.dumps(inspect()))
paths=[root/'supervised/supervisor.json',root/'supervised/run/run.json',root/'supervised/burn-verified.json',root/'supervised/run/complete.json',root/'supervised/gpus-free-before-training.json']
for backend in ('sdpa','fa4'):
 b=root/'supervised/run'/backend
 paths += [b/'training/train_results.json',b/'evaluation.json',b/'summary.json',b/'training/trainer_state.json',b/'checks/train.json',b/'checks/resume.json']
for p in paths:
 if p.exists():print('RESULT',json.dumps({'path':str(p),'data':json.loads(p.read_text())}))
for p in [root/'supervisor.log',root/'supervised/burn.log',*sorted((root/'supervised/run').glob('*.log'))]:
 if p.exists():
  lines=p.read_text(errors='replace').splitlines()
  print('LOG',str(p),'\n'.join(lines[-20:]))
  print('METRICS',str(p),'\n'.join(x for x in lines if x.startswith('{') and 'learning_rate' in x)[-6000:])
PY
