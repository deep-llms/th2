#1 +15+a
#th2-q359-p7-partial-links-20261015-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'PY'
import json,os,time
from pathlib import Path
from scripts.verified_gpu_reclaim import inspect
from scripts.train_then_burn import burn_progress
root=Path('/mnt/local/_outputs/deep-llms_th2')
p=root/'q359-p7-exploration8-20261015-a03/supervised/run/lm-noaux/seed-42/P7-simple-lm-noaux'
for f in p.rglob('*'):
 if f.is_symlink():print('LINK',str(f.relative_to(p)),os.readlink(f),'RESOLVES',str(f.resolve()))
for name in ['q359-p7-tuning22-20261015-a01','q359-p7-tuning22-20261015-a02']:
 print('OUTPUT_EXISTS',name,(root/name).exists())
print('INSPECT',json.dumps(inspect()))
f=root/'q359-p7-completed-eval-20261015-a02/supervised/burn.log'
print('BURN_PROGRESS',burn_progress(f.read_text()),'age',time.time()-f.stat().st_mtime)
PY
