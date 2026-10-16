#1 +15+a
#th2-q359-p7-priority-results-20261016-0730
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'PYSTATUS'
import json,math
from pathlib import Path
from datetime import datetime,timezone
from scripts.gpu_status import snapshot
base=Path('/mnt/local/_outputs/deep-llms_th2')
print('AT',datetime.now(timezone.utc).isoformat())
for name in ('q359-p7-tuning22-20261015-a03','q359-p7-priority15-20261015-a01'):
 root=base/name
 for rel in ('supervised/run/run.json','supervised/supervisor.json','supervised/burn-verified.json'):
  p=root/rel
  if p.exists():
   v=json.loads(p.read_text())
   if rel.endswith('/run.json'):v['jobs']=[j for j in v['jobs'] if not j['name'].startswith('smoke') and j['name']!='numerics']
   print('STATE',str(p),json.dumps(v))
 run=root/'supervised/run'
 for p in sorted(run.glob('*/seed-*/*/result.json')):
  v=json.loads(p.read_text()); e=v['evaluation']
  print('RESULT',json.dumps(dict(path=str(p),arm=v['arm'],step=v['global_step'],status=v['status'],evaluation=e,training_cost=v.get('training_cost'),validated=(p.parents[2]/'validated.json').exists())))
 state=json.loads((run/'run.json').read_text())
 if state['status']=='running':
  for job in state['jobs']:
   if job['status']=='running':
    p=run/(job['name']+'.log')
    if p.exists():print('ACTIVE_LOG',str(p),p.read_bytes()[-5500:].decode(errors='replace'))
p=base/'q359-proxy-12layer-screen22-20261013-a01/supervised/run/seed-42/A/result.json'
if p.exists():print('BASELINE',p.read_text())
print('GPUS',json.dumps(snapshot(list(range(8)))))
PYSTATUS
