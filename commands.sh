#1 +15+a
#th2-q359-p7-tuning-results-20261015-2223
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'STATUSPY'
import json
from pathlib import Path
from datetime import datetime,timezone
from scripts.gpu_status import snapshot
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-p7-tuning22-20261015-a03')
print('AT',datetime.now(timezone.utc).isoformat())
for name in ['supervised/supervisor.json','supervised/run/run.json','supervised/run/summary.json']:
 p=root/name
 if not p.exists():
  print('MISSING',name); continue
 v=json.loads(p.read_text())
 if name.endswith('/run.json'):
  print('QUEUE',v.get('status'),v.get('error'))
  for j in v.get('jobs',[]):
   if j['name'].startswith('smoke') or j['name']=='numerics':continue
   print('JOB',json.dumps({k:j.get(k) for k in ('name','status','started_at','finished_at')}))
   if j.get('status')=='running':
    log=root/'supervised/run'/(j['name']+'.log')
    if log.exists():
     with log.open('rb') as f:
      f.seek(max(0,log.stat().st_size-9000));print('ACTIVE_LOG',f.read().decode(errors='replace'))
 else: print('STATE',name,json.dumps(v))
for p in sorted((root/'supervised/run').glob('*/seed-*/*/result.json')):
 v=json.loads(p.read_text());print('RESULT',json.dumps(dict(path=str(p),arm=v.get('arm'),step=v.get('global_step'),evaluation=v.get('evaluation'),training_cost=v.get('training_cost'))))
p=Path('/mnt/local/_outputs/deep-llms_th2/q359-proxy-12layer-screen22-20261013-a01/supervised/run/seed-42/A/result.json')
if p.exists():print('BASELINE',p.read_text())
print('GPUS',json.dumps(snapshot(list(range(8)))))
STATUSPY
