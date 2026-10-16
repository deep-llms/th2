#1 +30+a
#th2-q359-ce7454-study-health-20261016-a05
set -euo pipefail
date -u
hostname
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv
/mnt/local/conda-py311/envs/eval_fa4/bin/python - <<'CHECK'
from pathlib import Path
import json,csv
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-ce7454-scratch-20261016-a04')
for name in ['supervised/supervisor.json','supervised/run/run.json','supervised/run/study/schedule.json','supervised/burn-verified.json']:
 p=root/name
 if p.exists(): print(str(p),p.read_text()[-5000:])
for p in sorted((root/'supervised/run/study').glob('*/log.csv')):
 rows=list(csv.DictReader(p.open()));print('EPOCHS',str(p),len(rows),rows[-5:])
for p in sorted((root/'supervised/run/study').glob('*.log')):
 if not p.name.startswith('benchmark'): print('LOG',p.name,p.read_text()[-1000:])
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
  if ('train_then_burn' in cmd or 'deadline_study.py' in cmd) and 'python' in cmd:
   print('OWNER',proc.name,(proc/'stat').read_text(),cmd)
 except (OSError,UnicodeError):pass
CHECK
