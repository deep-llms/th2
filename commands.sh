#0
#th2-q359-expanded-single-seed42-health-20261013-a02
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
python3 -u - <<'REMOTE'
import datetime,json,subprocess
from pathlib import Path
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-expanded-single-seed42-20261013-a01')
old=Path('/mnt/local/_outputs/deep-llms_th2/q359-expanded-paired-20k-20261012-a03')
print('UTC',datetime.datetime.now(datetime.timezone.utc).isoformat())
for p in (root/'preflight.json',root/'supervised/supervisor.json',root/'supervised/run/run.json',root/'supervised/run/finetune/run.json'):
 if p.exists():
  obj=json.loads(p.read_text())
  if 'jobs' in obj:obj={**obj,'jobs':obj['jobs'][-5:],'completed_count':sum(j['status']=='ok' for j in obj['jobs'])}
  print(str(p),json.dumps(obj),flush=True)
for base in (old,root):
 p=base/'supervised/run/finetune/run.json'
 if p.exists():
  for j in json.loads(p.read_text())['jobs']:
   if j['name'].startswith('search-'):print('FIT_TIMING',base.name,j['name'],j.get('status'),j.get('elapsed_seconds'))
p=Path(str(root)+'-launch.log')
print('LAUNCH_TAIL','\n'.join(p.read_text().splitlines()[-20:]))
fit=root/'supervised/run/finetune'
for p in sorted(fit.glob('*.log'),key=lambda x:x.stat().st_mtime)[-2:]:
 print('FIT_LOG',str(p),'\n'.join(p.read_text().splitlines()[-15:]))
subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv'],check=True)
REMOTE
