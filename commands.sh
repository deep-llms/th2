#1 +30+a
#th2-6y3o-a2-assets-inventory-20261028-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PY'
import json,socket,shutil,hashlib
from pathlib import Path
from datetime import datetime,timezone
from scripts.gpu_process import inspect
assert socket.gethostname()=='thiennh-p6-6y3o-worker-0'
print('INVENTORY_AT',datetime.now(timezone.utc).isoformat())
print('DISK',shutil.disk_usage('/mnt/local'))
print('GPU_INSPECTION',json.dumps(inspect()))
root=Path('/mnt/local/_data/deep-llms_th2/raw/cx_sampled_dd1b1513')
for rel in ('subsets/qwen3_0.6b_base_en_30B/manifest.json','completion.json'):
 p=root/rel
 print('POOL',str(p),p.exists())
 if p.is_file():
  b=p.read_bytes(); j=json.loads(b)
  print('POOL_MANIFEST',json.dumps(dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),keys=list(j),totals=j.get('totals'))))
  if 'shards' in j:
   for split,shards in j['shards'].items():
    missing=[];badsize=[];n=0;total=0
    for shard in shards:
     for name,meta in shard['files'].items():
      f=root/name;n+=1;total+=meta['bytes']
      if not f.is_file():missing.append(name)
      elif f.stat().st_size!=meta['bytes']:badsize.append(name)
    print('POOL_SPLIT',json.dumps(dict(split=split,shards=len(shards),files=n,bytes=total,missing=missing,badsize=badsize)))
for name in ('a2','Qwen3-0.6B-Base_da87bfb6'):
 p=Path('/mnt/local/_models/deep-llms_th2')/name
 print('MODEL_DIRECTORY',str(p),sorted(str(x.relative_to(p)) for x in p.rglob('*') if x.is_file()) if p.exists() else 'absent')
print('INVENTORY_COMPLETE')
PY
