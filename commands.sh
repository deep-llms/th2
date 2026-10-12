#1 +60+a
#th2-q359-paired-parallel-repair-health-20261013-a03
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
date -u
/mnt/local/conda-py311/envs/eval_fa4/bin/python -u - <<'REMOTE'
import json,time
from pathlib import Path
from collections import Counter
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-paired-parallel-repair-20261013-a02')
for attempt in range(6):
 state=root/'supervised/run/run.json';pool=root/'supervised/run/paired/parallel-status.json'
 status=json.loads(state.read_text()) if state.exists() else {}
 jobs=json.loads(pool.read_text()).get('jobs',[]) if pool.exists() else []
 print('WAIT_STATUS',time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),status.get('status'),dict(Counter(j.get('status') for j in jobs)),flush=True)
 if status.get('status') in ('failed','ok') or any(j['status']=='ok' and '/F/xstorycloze_en' in j['output'] for j in jobs):break
 time.sleep(45)
for rel in ['supervised/supervisor.json','supervised/run/run.json','restart-receipt.json','preflight.json','supervised/gpus-free-before-training.json','supervised/run/complete.json','supervised/burn-verified.json','supervised/run/paired/precision.json','supervised/run/paired-F.json','supervised/run/paired-Z.json','supervised/run/paired-probes.json','supervised/run/paired-position.json','supervised/run/paired/parallel-status.json']:
 p=root/rel
 print('FILE',rel,'EXISTS',p.exists())
 if not p.exists():continue
 value=json.loads(p.read_text())
 print('MTIME',time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime(p.stat().st_mtime)))
 if 'jobs' in value:
  jobs=value['jobs']; print('COUNTS',dict(Counter(j.get('status') for j in jobs)))
  if 'parallel-status' in rel:
   print('DEVICES',value.get('devices'));print('JOBS',json.dumps(jobs))
  else:
   value['jobs']=jobs[-5:];print(json.dumps(value))
 else:print(json.dumps(value)[:18000])
for p in [root.with_name(root.name+'-launch.log'),*sorted((root/'supervised/run').glob('paired*.log'))]:
 print('LOG',str(p))
 if p.exists():print(p.read_text(errors='replace')[-9000:])
p=root/'supervised/run/paired/parallel-status.json'
if p.exists():
 for job in json.loads(p.read_text())['jobs']:
  if job['status']!='running':continue
  q=Path(job['output']).with_suffix('.worker.log');print('WORKER_LOG',str(q))
  if q.exists():print(q.read_text(errors='replace')[-1800:])
REMOTE
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory --format=csv
