#1 +30+a
#th2-tjx3-proxy-remaining-monitor-20261006-a02
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
date -u
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'PY'
import json
from pathlib import Path
from scripts.gpu_status import snapshot
root=Path('/mnt/local/_outputs/deep-llms_th2/proxy-remaining-2500-20261006-a01');run=root/'supervised/run'
for name in ('launch.log','preflight.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/supervisor.json','supervised/burn-verified.json'):
 p=root/name
 if p.is_file():print('FILE',name,p.read_text()[-10000:],flush=True)
p=run/'run.json'
if p.is_file():
 state=json.loads(p.read_text());print('QUEUE',json.dumps(state),flush=True)
 for job in state['jobs'][-3:]:
  p=run/(job['name']+'.log')
  if p.is_file():print('JOB_LOG',job['name'],p.read_text()[-12000:],flush=True)
for p in sorted((run/'checks').glob('*.json')):
 d=json.loads(p.read_text());print('CHECK',p.name,json.dumps(d),flush=True)
for stage in ('smoke','training'):
 for folder in sorted((run/stage/'seed-42').glob('*')):
  p=folder/'trainer_state.json'
  if p.is_file():
   d=json.loads(p.read_text());print('TRAINER',stage,folder.name,d['global_step'],json.dumps(d['log_history'][-3:]),flush=True)
  p=folder/'result.json'
  if p.is_file():
   d=json.loads(p.read_text());print('RESULT',stage,folder.name,json.dumps({k:d[k] for k in ('status','global_step','training_cost','evaluation')}),flush=True)
print('GPUS',json.dumps(snapshot(list(range(8)))),flush=True)
print('MONITOR_FINISHED',flush=True)
PY
