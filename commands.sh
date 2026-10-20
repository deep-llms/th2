#1 +30+a
#th2-6y3o-corpus-transfer-status-20261020-a02
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
python3 - <<'CHECK'
from pathlib import Path
import json,datetime,hashlib,subprocess,re
root=Path('/mnt/local/_data/deep-llms_th2/raw/cx_sampled_dd1b1513')
sub=root/'subsets/qwen3_0.6b_base_en_30B'
m=json.loads((sub/'manifest.json').read_text())
groups={str(i):dict(files=0,bytes=0,newest_utc=None) for i in range(8)}
for split,shards in m['shards'].items():
 for s in shards:
  mat=re.search(r'/train/shard_(\d+)',s['path']);g=groups[str(int(mat[1])%8 if mat else 0)]
  for rel,meta in s['files'].items():
   f=root/rel
   if not f.is_file():continue
   st=f.stat()
   if st.st_size==meta['bytes']:
    g['files']+=1;g['bytes']+=st.st_size
    t=datetime.datetime.fromtimestamp(st.st_mtime,datetime.timezone.utc).isoformat()
    g['newest_utc']=max(g['newest_utc'] or '',t)
print('TRANSFER_FILES',json.dumps(dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),groups=groups)),flush=True)
contract=json.loads(Path('resources/cx_english_20261020.json').read_text())
model=Path('/mnt/local/_models/deep-llms_th2/Qwen3-0.6B-Base_da87bfb6')
for n,meta in contract['tokenizer_files'].items():
 p=model/n
 print('MODEL_ASSET',n,p.stat().st_size if p.is_file() else 'missing',p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==meta['sha256'],flush=True)
from scripts.gpu_process import inspect
print('GPU_OWNER',json.dumps(inspect()),flush=True)
CHECK
