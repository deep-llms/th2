#1 +15+a
#th2-q359-p7-tuning-eval-health-20261016-a02
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/eval_fa4/bin/python3.11 -u - <<'PYCHECK'
import json,time
from pathlib import Path
from datetime import datetime,timezone
from scripts.gpu_status import snapshot
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-p7-tuning-eval-20261016-a01')
source=Path('/mnt/local/_outputs/deep-llms_th2/q359-p7-priority15-20261015-a01')
for attempt in range(24):
 p=root/'supervised/run/run.json'
 if p.exists():
  v=json.loads(p.read_text());print('PROGRESS',datetime.now(timezone.utc).isoformat(),json.dumps(v),flush=True)
  if v['status']!='running' or any(j['name']=='full' for j in v['jobs']):break
 time.sleep(15)
print('AT',datetime.now(timezone.utc).isoformat())
for p in (source/'user-stop-for-evaluation.json',source/'supervised/run/run.json',root/'preflight.json',root/'scope.json',root/'supervised/supervisor.json',root/'supervised/gpus-free-before-training.json',root/'supervised/run/run.json',root/'supervised/burn-verified.json'):
 if p.exists():print('STATE',str(p),p.read_text())
p=Path(str(root)+'.launch.log')
if p.exists():print('LAUNCH',p.read_bytes()[-4000:].decode(errors='replace'))
for stage in ('numerics','smoke','full'):
 p=root/'supervised/run'/stage
 print('STAGE',stage,'reports',len(list(p.glob('*.json'))),'completed_groups',len([q for q in p.glob('*/*/eval_metadata.json') if json.loads(q.read_text()).get('status')=='completed']))
 q=p/'summary.json'
 if q.exists():print('SUMMARY',stage,q.read_text()[:1200])
 q=root/'supervised/run'/(stage+'.log')
 if q.exists():print('LOG',stage,q.read_bytes()[-2500:].decode(errors='replace'))
for p in (root/'supervised/run').rglob('*.log'):
 tail=p.read_bytes()[-2200:].decode(errors='replace')
 if 'Traceback' in tail or 'Error:' in tail:print('ERROR_LOG',str(p),tail)
print('GPUS',json.dumps(snapshot(list(range(8)))))
PYCHECK
