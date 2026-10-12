#1 +60+a
#th2-q359-paired-config-diff-20261013-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/eval_fa4/bin/python -u - <<'REMOTE'
import json
from pathlib import Path
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-expanded-single-seed42-20261013-a01/supervised/run/paired/precision')
def diff(a,b,path=''):
 if isinstance(a,dict) and isinstance(b,dict):
  for k in sorted(a.keys()|b.keys()):diff(a.get(k),b.get(k),path+'/'+k)
 elif a!=b:print(path,repr(a)[:1200],repr(b)[:1200])
for task in sorted((root/'reference/A20').iterdir()):
 name=task.name
 if not task.is_dir():continue
 print('TASK',name)
 a=json.loads((task/'eval_benchmarks_full.json').read_text())['configs']
 for mode in ('reference','production','repeat'):
  for arm in ('A20','P20'):
   b=json.loads((root/mode/arm/name/'eval_benchmarks_full.json').read_text())['configs']
   print('COMPARE',mode,arm);diff(a,b)
REMOTE
