#1 +15+a
#th2-q359-p7-tuning-eval-health-20261016-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/eval_fa4/bin/python3.11 -u - <<'PYCHECK'
import json
from pathlib import Path
from datetime import datetime,timezone
from scripts.gpu_status import snapshot
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-p7-tuning-eval-20261016-a01')
source=Path('/mnt/local/_outputs/deep-llms_th2/q359-p7-priority15-20261015-a01')
print('AT',datetime.now(timezone.utc).isoformat())
for p in (source/'user-stop-for-evaluation.json',source/'supervised/run/run.json',root/'preflight.json',root/'scope.json',root/'supervised/supervisor.json',root/'supervised/gpus-free-before-training.json',root/'supervised/run/run.json',root/'supervised/burn-verified.json'):
 if p.exists():print('STATE',str(p),p.read_text())
p=Path(str(root)+'.launch.log')
if p.exists():print('LAUNCH',p.read_bytes()[-4000:].decode(errors='replace'))
for stage in ('numerics','smoke','full'):
 p=root/'supervised/run'/stage
 print('STAGE',stage,'reports',len(list(p.glob('*.json'))),'completed_groups',len(list(p.glob('*/completion.json'))))
 q=p/'summary.json'
 if q.exists():print('SUMMARY',stage,q.read_text()[:1200])
 q=root/'supervised/run'/(stage+'.log')
 if q.exists():print('LOG',stage,q.read_bytes()[-2500:].decode(errors='replace'))
print('GPUS',json.dumps(snapshot(list(range(8)))))
PYCHECK
