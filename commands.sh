#1 +30+a
#th2-q359-template-smoke-review-20261017-a02
set -euo pipefail
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/eval_fa4/bin/python -B - <<'PY'
import json
from pathlib import Path
from datetime import datetime,timezone
from scripts.verified_gpu_reclaim import inspect
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-template-smoke-20261017-a03')
print('TIME',datetime.now(timezone.utc).isoformat())
print('INSPECTION',json.dumps(inspect()))
for p in [root/'supervised/supervisor.json',root/'supervised/run/run.json',root/'supervised/run/fa4-kernel.json',root/'supervised/burn-verified.json',*sorted((root/'supervised/run').glob('*/complete.json'))]:
 if p.exists():print('RESULT',str(p),p.read_text())
for p in [root/'supervisor.log',root/'supervised/burn.log',*sorted((root/'supervised/run').rglob('*.log'))]:
 if p.exists():print('LOG',str(p),'\n'.join(p.read_text(errors='replace').splitlines()[-18:]))
PY
