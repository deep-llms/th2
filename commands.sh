#1 +15+a
#th2-q359-teacher-first-health-20261016-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/eval_fa4/bin/python3.11 -u - <<'PY'
import json,time
from pathlib import Path
from datetime import datetime,timezone
from scripts.gpu_status import snapshot
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-teacher-first-20261016-a01')
print('AT',datetime.now(timezone.utc).isoformat())
for name in ('preflight.json','supervised/supervisor.json','supervised/run/run.json','supervised/run/summary.json','supervised/burn-verified.json'):
 p=root/name
 if p.exists():print('STATE',name,p.read_text())
for p in [root.with_suffix('.launch.log'),*sorted((root/'supervised/run').glob('*.log'))]:
 if p.exists():print('LOG',str(p),p.read_bytes()[-5000:].decode(errors='replace'))
for p in sorted((root/'supervised/run').glob('*/initial.json')):print('NUMERICS',str(p),p.read_text())
print('GPUS',json.dumps(snapshot(list(range(8)))))
PY
