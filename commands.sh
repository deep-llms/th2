#1 +15+a
#th2-q359-p7-priority-verify-20261015-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'PYVERIFY'
import json,time
from pathlib import Path
from datetime import datetime,timezone
from scripts.gpu_status import snapshot
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-p7-priority15-20261015-a01')
old=Path('/mnt/local/_outputs/deep-llms_th2/q359-p7-tuning22-20261015-a03')
for attempt in range(30):
 p=root/'supervised/run/run.json'
 if p.exists():
  v=json.loads(p.read_text());print('NEW_QUEUE',v,flush=True)
  if v['status']!='running' or (v['jobs'] and time.time()-p.stat().st_mtime>45):break
 print('WAIT',datetime.now(timezone.utc).isoformat(),flush=True)
 time.sleep(20)
print('AT',datetime.now(timezone.utc).isoformat())
for base in (old,root):
 for name in ('request.json','boundary-installed.json','handoff-verified.json','supervised/supervisor.json','supervised/burn-verified.json','supervised/gpus-free-before-training.json','supervised/run/run.json'):
  p=base/name
  if p.exists():
   v=json.loads(p.read_text())
   if name.endswith('/run.json'):v['jobs']=[j for j in v['jobs'] if not j['name'].startswith('smoke') and j['name']!='numerics']
   print('STATE',str(p),json.dumps(v))
 log=Path(str(base)+'.launch.log')
 if log.exists():print('LAUNCH',str(log),log.read_bytes()[-7000:].decode(errors='replace'))
p=root/'supervised/run/tg-valueonly-train.log'
if p.exists():print('TRAIN',p.read_bytes()[-6000:].decode(errors='replace'))
print('GPUS',json.dumps(snapshot(list(range(8)))))
PYVERIFY
