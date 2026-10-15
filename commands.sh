#1 +15+a
#th2-q359-p7-tuning-inspect-20261015-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'REMOTE'
import json,shutil,time
from pathlib import Path
from datetime import datetime,timezone
from scripts.verified_gpu_reclaim import inspect
from scripts.train_then_burn import burn_progress
base=Path('/mnt/local/_outputs/deep-llms_th2')
old=base/'q359-p7-exploration8-20261015-a03'
previous=base/'q359-p7-completed-eval-20261015-a02'
print('AT',datetime.now(timezone.utc).isoformat())
print('FREE_GIB',shutil.disk_usage(base).free/2**30)
for root,names in ((old,('user-stop-after-keyonly.json','supervised/supervisor.json','supervised/run/run.json')),(previous,('supervised/supervisor.json','supervised/run/complete.json','supervised/burn-verified.json'))):
 for name in names:
  p=root/name;print('STATE',str(p),p.read_text() if p.exists() else 'MISSING')
for label,arm in [('lm-noaux','P7-simple-lm-noaux'),('lm-nexttoken','P7-simple-lm-nexttoken')]:
 p=old/'supervised/run'/label
 print('LEFTOVER',str(p),'exists',p.exists(),'symlink',p.is_symlink())
 if p.exists():
  print('FILES',[str(x.relative_to(p)) for x in p.rglob('*') if x.is_file() and x.suffix=='.json'])
  for name in ['validated.json',f'seed-42/{arm}/train_config.json',f'seed-42/{arm}/result.json']:
   f=p/name
   if f.exists():print('DETAIL',name,f.read_text())
p=previous/'supervised/burn.log'
print('BURN',burn_progress(p.read_text()),'age',time.time()-p.stat().st_mtime)
print('INSPECTION',json.dumps(inspect()))
REMOTE
