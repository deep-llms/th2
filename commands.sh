#1 +15+a
#th2-q359-p7-exploration-status-20261015-1218
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'REMOTE'
import json,time
from pathlib import Path
from datetime import datetime,timezone
from scripts.gpu_status import snapshot
from scripts.train_then_burn import burn_progress,approved_launcher,process,BURN_HASH
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-p7-exploration8-20261015-a03')
run=root/'supervised/run'
print('AT',datetime.now(timezone.utc).isoformat(),flush=True)
for name in ('supervised/supervisor.json','supervised/run/run.json','supervised/run/summary.json','supervised/burn-verified.json'):
 p=root/name
 print('STATE',name,p.read_text() if p.exists() else 'MISSING')
for label in ('a-long','targetgrad-long','lm-long','lm-sparse','lm-keyonly','lm-valueonly','lm-noaux','lm-nexttoken'):
 base=run/label/'seed-42'
 for folder in base.glob('*'):
  if not folder.is_dir():continue
  ck=sorted(int(p.name.split('-')[-1]) for p in folder.glob('checkpoint-*') if p.is_dir())
  print('CHECKPOINTS',label,ck)
  for name in ('result.json','trainer_state.json'):
   p=folder/name
   if p.exists():
    v=json.loads(p.read_text())
    if name=='result.json':v={k:v[k] for k in ('arm','global_step','status','evaluation','training_cost') if k in v}
    if name=='trainer_state.json':v={k:v[k] for k in ('global_step','max_steps','log_history') if k in v};v['log_history']=v.get('log_history',[])[-5:]
    print('ARM_STATE',label,name,json.dumps(v))
  if ck:
   p=folder/f'checkpoint-{ck[-1]}'/'trainer_state.json'
   if p.exists():
    v=json.loads(p.read_text());print('LATEST_SAVED',label,v['global_step'],json.dumps(v['log_history'][-3:]))
 p=run/(label+'-train.log')
 if p.exists():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-14000));tail=f.read().decode(errors='replace')
  print('LOG_TAIL',label,'age_seconds',time.time()-p.stat().st_mtime,tail)
print('GPU_SNAPSHOT',json.dumps(snapshot(list(range(8)))))
p=root/'supervised/burn.log'
if p.exists():print('BURN_PROGRESS',burn_progress(p.read_text()),'log_age_seconds',time.time()-p.stat().st_mtime)
print('COMPLETE',(run/'complete.json').exists())
REMOTE
