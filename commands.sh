#0
#th2-tjx3-stsb-boolq-monitor-20261008-a02
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
date -u
/mnt/local/conda-py311/envs/eval_fa4/bin/python -u - <<'PY'
import hashlib,json
from pathlib import Path
from scripts.gpu_status import snapshot
root=Path('/mnt/local/_outputs/deep-llms_th2/supervised-stsb-boolq-20261008-a01');run=root/'supervised/run'
def artifact(p,name):
 if p.is_file():
  raw=p.read_bytes();print('ARTIFACT',json.dumps(dict(name=name,sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode())),flush=True)
p=root/'launch.log'
if p.is_file():print('LAUNCH_LOG',p.read_text()[-7000:],flush=True)
for name in ('data-audit.json','preflight.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/supervisor.json','supervised/burn-verified.json','supervised/run/complete.json'):
 artifact(root/name,name)
p=run/'run.json'
if p.is_file():
 state=json.loads(p.read_text());print('QUEUE',json.dumps(state),flush=True)
 for job in state['jobs'][-3:]:
  p=run/(job['name']+'.log')
  if p.is_file():print('JOB_LOG',job['name'],p.read_text()[-7000:],flush=True)
for p in sorted((run/'gates').glob('*.json')):artifact(p,'gates/'+p.name)
for p in sorted((run/'smoke-checks').glob('*.json')):artifact(p,'smoke-checks/'+p.name)
for p in sorted((run/'selections').glob('*.json')):artifact(p,'selections/'+p.name)
for phase in ('smoke','smoke-eval','search','confirm','test'):
 for p in sorted((run/phase).rglob('result.json')):artifact(p,str(p.relative_to(run)))
artifact(run/'summary.json','summary.json')
p=root/'supervised/burn.log'
if p.is_file():
 with p.open('rb') as stream:
  stream.seek(max(0,p.stat().st_size-5000));print('BURN_LOG',stream.read().decode(errors='replace'),flush=True)
print('GPUS',json.dumps(snapshot(list(range(8)))),flush=True)
print('MONITOR_FINISHED',flush=True)
PY
