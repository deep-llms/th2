#1 +10+a
#th2-ohmv-anchor-long-health-20261122-a01
set -euo pipefail
/usr/bin/python3 - <<'PY'
from pathlib import Path
import json,time
root=Path('/mnt/local/_outputs/deep-llms_th2')
base=root/'anchor_long_20261122_a01'
r={'sample_epoch':time.time(),'long':{},'workers':{},'cpu':{}}
for n in ('status.json','failure.json','terminal.json','rows/manifest.json'):
 p=base/n
 if p.is_file():r['long'][n]=json.loads(p.read_text())
for seed in (0,1):
 for arm in ('shared','direct','additive','reader'):
  for n in ('progress.json','complete.json','passed.json'):
   p=base/f's{seed}'/(('qualification-'+arm) if n=='passed.json' else arm)/n
   if p.is_file():r['workers'][str(p.relative_to(base))]=json.loads(p.read_text())
  for stage in ('fit','qualify'):
   p=base/f'{stage}-s{seed}-{arm}.log'
   if p.is_file() and (base/'failure.json').exists():r['workers'][p.name]=p.read_text()[-1500:]
for p in (base/'execution/queue.log',):
 if p.exists():r['long'][p.name]=p.read_text()[-900:]
p=root/'anchor_boundary_cpu_20261122_a01/report.json'
if p.is_file():r['cpu']=json.loads(p.read_text())
print('ANCHOR_LONG_HEALTH',json.dumps(r,allow_nan=False))
PY
