#1 +15+a
#th2-q359-p7-tuning-eval-results-20261016-0947
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/eval_fa4/bin/python3.11 -u - <<'PYSTATUS'
import json,time,hashlib
from pathlib import Path
from datetime import datetime,timezone
from scripts.gpu_status import snapshot
from scripts.train_then_burn import burn_progress
from scripts.verified_gpu_reclaim import ownership
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-p7-tuning-eval-20261016-a01')
print('AT',datetime.now(timezone.utc).isoformat())
for name in ('supervised/supervisor.json','supervised/run/run.json','supervised/run/complete.json','supervised/burn-verified.json'):
 p=root/name
 if p.exists():print('STATE',name,json.dumps(json.loads(p.read_text())))
p=root/'supervised/run/full/summary.json'
if p.exists():
 print('SUMMARY_SHA256',hashlib.sha256(p.read_bytes()).hexdigest())
 print('SUMMARY',json.dumps(json.loads(p.read_text())))
for stage in ('numerics','smoke','full'):
 p=root/'supervised/run'/stage
 print('STAGE',stage,'completed_groups',len([q for q in p.glob('*/*/eval_metadata.json') if json.loads(q.read_text()).get('status')=='completed']))
q=root/'supervised/run/run.json';state=json.loads(q.read_text())
if state['status']=='failed':
 for p in (root/'supervised/run').rglob('*.log'):
  tail=p.read_bytes()[-2500:].decode(errors='replace')
  if 'Traceback' in tail or 'Error:' in tail:print('ERROR_LOG',str(p),tail)
gpus=snapshot(list(range(8)));print('GPUS',json.dumps(gpus))
p=root/'supervised/burn.log';b=root/'supervised/burn-verified.json'
if p.exists() and b.exists():
 text=p.read_text();burn=json.loads(b.read_text())
 print('LIVE_BURN',json.dumps(dict(progress=burn_progress(text),age_seconds=time.time()-p.stat().st_mtime,ownership_matches=ownership(gpus)==ownership(burn['gpus']),guard_disabled=Path('/mnt/local/_gpu_guard/DISABLED').exists())))
 print('BURN_TAIL',text[-2500:])
PYSTATUS
