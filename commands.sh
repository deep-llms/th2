#1 +120+a
#th2-6y3o-quant-replay-package-20261031-b06
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
python3 -u - <<'PY'
"""Lossless transport of existing replay JSON; reconstruct and SHA-check exact original bytes."""
import gzip,hashlib,json
from pathlib import Path
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01')
out=root/'retrieval_20261031_b06';out.mkdir(exist_ok=False)
m=json.loads((root/'retrieval_20261031_b04/manifest.json').read_text())
files=m['groups']['replays']['files']
shared=None;records=[];bundle=out/'replays_shared.jsonl.gz'
with gzip.open(bundle,'wt',encoding='utf-8',compresslevel=3) as stream:
 for row in files:
  raw=Path(row['path']).read_bytes()
  assert hashlib.sha256(raw).hexdigest()==row['sha256']
  value=json.loads(raw)
  encoding=None
  for style in ('indent2','plain'):
   encoded=(json.dumps(value,indent=2,allow_nan=False) if style=='indent2' else json.dumps(value,allow_nan=False))+'\n'
   if hashlib.sha256(encoded.encode()).hexdigest()==row['sha256']:encoding=style;break
  assert encoding is not None,'Unrecognized JSON serialization; retain original: '+row['path']
  del raw,encoded
  if shared is None:
   shared={task:{str(s['doc_id']):{k:s[k] for k in ('doc','arguments','target')} for s in samples} for task,samples in value['samples'].items()}
   stream.write(json.dumps({'type':'shared_prompts','value':shared},allow_nan=False)+'\n')
  for task,samples in value['samples'].items():
   assert len(samples)==len(shared[task])
   for s in samples:
    ref=shared[task][str(s['doc_id'])]
    for k in ('doc','arguments','target'):
     assert s[k]==ref[k],(row['member'],task,s['doc_id'],k)
     s[k]=None
  entry={'type':'replay','member':row['member'],'sha256':row['sha256'],'bytes':row['bytes'],'encoding':encoding,'value':value}
  stream.write(json.dumps(entry,allow_nan=False)+'\n')
  records.append({k:v for k,v in entry.items() if k!='value'})
  print('DEDUP_FILE',row['member'],flush=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
parts=[]
with bundle.open('rb') as f:
 i=0
 while True:
  data=f.read(16*1024*1024)
  if not data:break
  p=out/('replays_shared.part'+str(i).zfill(3));p.write_bytes(data)
  parts.append({'path':str(p),'bytes':len(data),'sha256':sha(p)});i+=1
manifest={'format':'a2_replay_shared_prompts_v1','files':records,'archive_sha256':sha(bundle),'archive_bytes':bundle.stat().st_size,'parts':parts,'original_manifest_sha256':sha(root/'retrieval_20261031_b04/manifest.json')}
p=out/'manifest.json';p.write_text(json.dumps(manifest,indent=2)+'\n')
print('DEDUP_COMPLETE',json.dumps({'manifest':{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size},'file_count':len(records),'archive_bytes':bundle.stat().st_size,'parts':parts}),flush=True)

PY
