#1 +60+a
#th2-ohmv-embedding-anchor-qualification-health-20261122-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 - <<'PY'
import json,datetime
from pathlib import Path
from efficient_llm.warm_order.data import sha
from scripts.gpu_process import inspect
base=Path('/mnt/local/_outputs/deep-llms_th2/embedding_anchor_gpu_20261122_a01')
r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),base_exists=base.exists(),ownership=inspect(),seeds={})
for name in ('status.json','failure.json','terminal.json','qualification-complete.json','execution/submitted.json'):
 p=base/name
 if p.exists():r[name]=json.loads(p.read_text())
p=base/'execution/source.json'
if p.exists():
 pins=json.loads(p.read_text())
 r['source']=dict(count=len(pins),all_match=all(sha(base/'execution/source'/n)==h for n,h in pins.items()),sha256=sha(p))
for seed in (0,1):
 root=base/f's{seed}';d={}
 for name in ('complete.json','numerical.json'):
  p=root/name
  if p.exists():d[name]=json.loads(p.read_text());d[name+'_sha256']=sha(p)
 p=base/f'study-s{seed}.log'
 if p.exists():d['log_tail']=p.read_text()[-4000:]
 r['seeds'][str(seed)]=d
p=base/'execution/queue.log'
if p.exists():r['queue_tail']=p.read_text()[-2000:]
p=base/'backup.json'
if p.exists():
 b=json.loads(p.read_text());r['backup']=dict(sha256=sha(p),archive_sha256=b['archive_sha256'],archive_bytes=b['archive_bytes'],copies=len(b['node_only_copies']),production_updates=b['production_updates'])
p=base/'native-verified.json'
if p.exists():r['native']=dict(sha256=sha(p),proof=json.loads(p.read_text()))
print('ANCHOR_QUALIFICATION_HEALTH',json.dumps(r),flush=True)
PY
