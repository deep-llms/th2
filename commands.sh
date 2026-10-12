#1 +60+a
#th2-q359-paired-completion-check-20261013-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
date -u
/mnt/local/conda-py311/envs/eval_fa4/bin/python -u - <<'REMOTE'
import json,hashlib
from pathlib import Path
from collections import Counter
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-paired-parallel-repair-20261013-a02');run=root/'supervised/run'
for rel in ['supervised/supervisor.json','supervised/burn-verified.json','supervised/run/run.json','supervised/run/complete.json','supervised/run/paired/parallel-status.json']:
 p=root/rel;print('FILE',rel,'EXISTS',p.exists())
 if not p.exists():continue
 v=json.loads(p.read_text())
 if 'parallel-status' in rel:
  print('COUNTS',dict(Counter(j['status'] for j in v['jobs'])));v['jobs']=[j for j in v['jobs'] if j['status']!='ok']+v['jobs'][-3:]
 print(json.dumps(v))
for p in [run/'paired-F.json',run/'paired-Z.json',run/'paired-probes.json',run/'paired-position.json']:
 print('REPORT',p.name,'EXISTS',p.exists())
 if p.exists():
  print('HASH',hashlib.sha256(p.read_bytes()).hexdigest(),'BYTES',p.stat().st_size)
  v=json.loads(p.read_text());print(json.dumps({k:x for k,x in v.items() if k in ('averages','confirmatory','progress_ratios','pairs','difference_of_profiles')}))
state=json.loads((run/'run.json').read_text());last=state['jobs'][-1]['name']
for p in [root.with_name(root.name+'-launch.log'),run/(last+'.log')]:
 print('LOG',str(p));print(p.read_text(errors='replace')[-5000:] if p.exists() else 'MISSING')
for p in (root/'supervised').glob('*burn*.log'):
 print('BURNLOG',p.name);print(p.read_text(errors='replace')[-3500:])
REMOTE
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory --format=csv
