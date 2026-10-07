#1 +30+a
#th2-tjx3-downstream-monitor-20261007-a04
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
date -u
/mnt/local/conda-py311/envs/eval_fa4/bin/python -u - <<'PY'
import hashlib,json
from pathlib import Path
from scripts.gpu_status import snapshot
root=Path('/mnt/local/_outputs/deep-llms_th2/downstream-2500-20261007-a01');run=root/'supervised/run'
def artifact(p,name):
 if p.is_file():
  raw=p.read_bytes();print('ARTIFACT',json.dumps(dict(name=name,sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode())),flush=True)
print('LAUNCH_LOG',(root/'launch.log').read_text()[-5500:],flush=True)
for name in ('data-validation.json','preflight.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/supervisor.json','supervised/burn-verified.json','supervised/run/complete.json'):
 artifact(root/name,name)
p=run/'run.json'
if p.is_file():
 state=json.loads(p.read_text());print('QUEUE',json.dumps(state),flush=True)
 for job in state['jobs'][-2:]:
  p=run/(job['name']+'.log')
  if p.is_file():print('JOB_LOG',job['name'],p.read_text()[-4000:],flush=True)
for p in sorted((run/'checks').glob('*.json')):artifact(p,'checks/'+p.name)
for name in ('smoke-validation.json','full-validation.json'):artifact(run/name,name)
for phase in ('smoke','full'):
 p=run/phase/'checkpoints.json';artifact(p,phase+'/checkpoints.json')
 if p.is_file():
  for path in json.loads(p.read_text()).values():
   folder=Path(path)
   for name in ('eval_metadata.json','eval_benchmarks.json'):
    artifact(folder/name,phase+'/'+folder.name+'/'+name)
   p=folder/'eval.log'
   if p.is_file():print('EVAL_LOG',phase,folder.name,p.read_text()[-1300:],flush=True)
print('GPUS',json.dumps(snapshot(list(range(8)))),flush=True)
print('MONITOR_FINISHED',flush=True)
PY
