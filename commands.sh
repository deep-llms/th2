#1 +15+a
#th2-q359-p7-tuning-health-20261015-a09
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'PY'
import json,time
from pathlib import Path
from datetime import datetime,timezone
from scripts.gpu_status import snapshot
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-p7-tuning22-20261015-a03')
print('AT',datetime.now(timezone.utc).isoformat())
for name in ['preflight.json','supervised/supervisor.json','supervised/gpus-free-before-training.json','supervised/run/numerics/summary.json','supervised/run/smoke-recipes.json','supervised/run/cleanup.json']:
 p=root/name
 if p.exists():
  v=json.loads(p.read_text()); print('STATE',name,{k:x for k,x in v.items() if k not in ('cases','gpus','versions')},'case_count',len(v.get('cases',[])))
 else:print('STATE',name,'MISSING')
p=root/'supervised/run/run.json'
if p.exists():
 v=json.loads(p.read_text());print('QUEUE_STATE',v.get('status'),v.get('error'))
 for j in v.get('jobs',[]):print('JOB',j.get('name'),j.get('status'),j.get('started_at'),j.get('finished_at'))
 for j in v.get('jobs',[]):
  if j.get('status') not in ['ok','pending']:
   p=root/'supervised/run'/(j['name']+'.log')
   if p.exists():print('JOB_LOG',j['name'],p.read_bytes()[-4500:].decode(errors='replace'))
for p in (root/'supervised/run/numerics').rglob('*.log'):
 if any(x in p.read_text(errors='replace') for x in ('Traceback','AssertionError')):print('ERROR_LOG',p.name,p.read_bytes()[-7000:].decode(errors='replace'))
for p in (root/'supervised/run').glob('*/seed-*/*/trainer_state.json'):
 v=json.loads(p.read_text());print('TRAIN_STATE',str(p),v.get('global_step'),v.get('log_history',[])[-2:])
p=Path(str(root)+'.launch.log')
if p.exists():print('LAUNCH_LOG',p.read_bytes()[-3000:].decode(errors='replace'))
old=Path('/mnt/local/_outputs/deep-llms_th2/q359-p7-exploration8-20261015-a03/supervised/run')
print('OLD_PARTIAL_EXISTS',(old/'lm-noaux/seed-42/P7-simple-lm-noaux').exists())
for label,arm in [('a-long','A'),('targetgrad-long','P7-simple-targetgrad'),('lm-long','P7-simple-lm'),('lm-sparse','P7-simple-lm-sparse'),('lm-keyonly','P7-simple-lm-keyonly'),('lm-valueonly','P7-simple-lm-valueonly')]:
 print('COMPLETED_OLD_PRESERVED',label,(old/label/'seed-42'/arm/'result.json').is_file())
print('GPUS',json.dumps(snapshot(list(range(8)))))
PY
