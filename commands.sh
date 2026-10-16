#1 +30+a
#th2-q359-ce7454-eight-health-20261016-a01
set -euo pipefail
date -u
hostname
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
/mnt/local/conda-py311/envs/eval_fa4/bin/python - <<'CHECK'
from pathlib import Path
import json,csv
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-ce7454-band-20261016-a01')
for rel in ['controller.json','preflight.json','gate_patch.json','released.json','study.log','study/schedule.json','study/complete.json']:
 p=root/rel
 if p.exists(): print('EXTENSION',rel,p.read_text()[-7000:])
for p in sorted((root/'study').glob('*.log')):
 print('LOG',p.name,p.read_text()[-2200:])
for p in sorted((root/'study').glob('smoke_*/result.json')):
 d=json.loads(p.read_text());d.pop('losses',None);print('NUMERICS',p.parent.name,d)
for p in sorted((root/'study').glob('*/log.csv')):
 rows=list(csv.DictReader(p.open()));print('EPOCH',p.parent.name,len(rows),rows[-2:])
old=Path('/mnt/local/_outputs/deep-llms_th2/q359-ce7454-scratch-20261016-a04')
print('ORIGINAL',(old/'supervised/supervisor.json').read_text())
for p in sorted((old/'supervised/run/study').glob('*/log.csv')):
 rows=list(csv.DictReader(p.open()));print('ORIGINAL_EPOCH',p.parent.name,len(rows),rows[-2:])
CHECK
