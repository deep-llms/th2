#0
#th2-q359-proxy12-eval-eta-20261014-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/eval_fa4/bin/python3.11 -u - <<'PY'
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-proxy12-eval-20261014-a01')
print('AT',datetime.now(timezone.utc).isoformat())
for name in ('supervised/supervisor.json','supervised/run/run.json'):
 p=root/name
 if p.exists():print('STATUS',name,p.read_text())
def tail(p,n=5):
 if not p.exists():return
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-16000));s=f.read().decode(errors='replace')
 print('TAIL',str(p),'\n'+'\n'.join(s.splitlines()[-n:]))
full=root/'supervised/run/full'
plan=full/'plan.json'
if plan.exists():
 for j in json.loads(plan.read_text()):
  dest=Path(j['output']);meta=dest/'eval_metadata.json';log=dest.with_suffix('.log')
  state=json.loads(meta.read_text()) if meta.exists() else {}
  print('JOB',j['checkpoint'].split('/')[-2],j['group'],state.get('status','started' if log.exists() else 'pending'),state.get('elapsed_seconds'), 'log_mtime',log.stat().st_mtime if log.exists() else None)
  if log.exists():tail(log,4)
for name in ('supervisor.log','supervised/run/full.log','supervised/burn.log'):tail(root/name)
if (full/'summary.json').exists():print('SUMMARY',json.loads((full/'summary.json').read_text())['status'])
subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv'])
PY
