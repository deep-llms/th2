#1 +30+a
#th2-tjx3-proxy-queue-status-20261007-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
date -u
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'REMOTE'
import json
from pathlib import Path
from scripts.gpu_status import snapshot
root=Path('/mnt/local/_outputs/deep-llms_th2/proxy-remaining-2500-20261006-a01')
run=root/'supervised/run'
state=json.loads((run/'run.json').read_text())
print('QUEUE_STATUS',state['status'],flush=True)
for job in state['jobs']:
 if job['name'].startswith('training-seed-42-arm-') or job['status']!='ok':
  print('STAGE',json.dumps({k:v for k,v in job.items() if k!='artifacts'}),flush=True)
last=state['jobs'][-1]
p=run/(last['name']+'.log')
if p.is_file():print('ACTIVE_LOG',last['name'],p.read_text()[-6000:],flush=True)
for arm in ('P7-simple','P7','P4-iso','P6','P5','P7-mlp','P7-kq','P7-ems','P4','P6-iso'):
 folder=run/'training/seed-42'/arm
 result=folder/'result.json'
 if result.is_file():
  d=json.loads(result.read_text());print('RESULT',arm,d['global_step'],d['evaluation']['eval_lm_loss'],flush=True)
 else:
  checkpoints=sorted(folder.glob('checkpoint-*/trainer_state.json'),key=lambda p:int(p.parent.name.split('-')[-1]))
  if checkpoints:
   d=json.loads(checkpoints[-1].read_text());print('CHECKPOINT',arm,d['global_step'],flush=True)
for name in ('supervised/supervisor.json','supervised/burn-verified.json','supervised/run/complete.json'):
 p=root/name
 if p.is_file():print('FILE',name,p.read_text(),flush=True)
print('GPUS',json.dumps(snapshot(list(range(8)))),flush=True)
print('MONITOR_FINISHED',flush=True)
REMOTE
