#1 +15+a
#th2-q359-screen22-queue-20261013-a06
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'PYREMOTE'
import json
from datetime import datetime,timezone
from pathlib import Path
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-proxy-12layer-screen22-20261013-a01');run=root/'supervised/run'
print('AT',datetime.now(timezone.utc).isoformat(),flush=True)
x=json.loads((run/'run.json').read_text());print('STATUS',x['status'],flush=True)
for j in x['jobs']:
 if j['name'].startswith('seed-42-'):print('JOB',json.dumps(j),flush=True)
 if j['status']=='running':
  p=run/(j['name']+'.log')
  if p.exists():
   with p.open('rb') as f:
    f.seek(max(0,p.stat().st_size-30000));s=f.read().decode(errors='replace')
   print('CURRENT_LOG',j['name'],'\n'+'\n'.join(s.splitlines()[-8:]),flush=True)
print('SUPERVISOR',(root/'supervised/supervisor.json').read_text(),flush=True)
PYREMOTE
