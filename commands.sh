#1 +15+a
#th2-q359-p7-eval-eta-20261015-1553
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'REMOTE'
import json,time
from pathlib import Path
from datetime import datetime,timezone
from scripts.gpu_status import snapshot
from scripts.train_then_burn import burn_progress
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-p7-completed-eval-20261015-a02')
print('AT',datetime.now(timezone.utc).isoformat(),flush=True)
for name in ('supervised/supervisor.json','supervised/run/run.json','supervised/run/complete.json','supervised/burn-verified.json'):
 p=root/name;print('STATE',name,p.read_text() if p.exists() else 'MISSING')
run=root/'supervised/run/full'
for i in range(8):
 for group in ('grammar','zero','five','ten','twentyfive'):
  folder=run/f'{i:02d}'/group;p=folder/'eval_metadata.json';v=json.loads(p.read_text()) if p.exists() else {}
  print('GROUP',i,group,v.get('status','not_started'),'modified',datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat() if p.exists() else None)
  log=folder.with_suffix('.log')
  if log.exists() and v.get('status')!='completed':
   with log.open('rb') as f:f.seek(max(0,log.stat().st_size-3000));print('ACTIVE_LOG',i,group,'age',time.time()-log.stat().st_mtime,f.read().decode(errors='replace'))
p=run/'summary.json'
if p.exists():
 v=json.loads(p.read_text());print('FINAL_SUMMARY',json.dumps(v))
print('GPUS',json.dumps(snapshot(list(range(8)))))
p=root/'supervised/burn.log'
if p.exists():print('BURN_PROGRESS',burn_progress(p.read_text()),'age',time.time()-p.stat().st_mtime)
REMOTE
