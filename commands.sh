#1 +15+a
#th2-q359-p7-tuning-health-20261015-a04
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
from deep_kv.report import comparable_config
from scripts.p7_tuning_queue import ALL_CASES
old=Path('/mnt/local/_outputs/deep-llms_th2/q359-p7-exploration8-20261015-a03/supervised/run/lm-noaux/seed-42/P7-simple-lm-noaux/train_config.json')
def norm(p):
 v=comparable_config(json.loads(p.read_text()))
 for k in ('logging_steps','save_steps','eval_steps','eval_on_start'):v['training'].pop(k,None)
 for k in ('eval_rows','monitor_rows'):v['data'].pop(k,None)
 v.pop('eval_fingerprint',None);v['pilot']['proxy_lambda_max']=.1
 return v
if old.exists():
 expected=norm(old)
 for label,arm,_,seed in ALL_CASES:
  p=root/'supervised/run/smoke'/label/f'seed-{seed}'/arm/'train_config.json'
  if seed==42 and p.exists():
   actual=norm(p);print('EARLY_REFERENCE_MATCH',label,actual==expected)
   if actual!=expected:print('DIFF',{k:(expected.get(k),v) for k,v in actual.items() if expected.get(k)!=v})
print('GPUS',json.dumps(snapshot(list(range(8)))))
PY
