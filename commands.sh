#1 +20+a
#th2-wc69-backup-input-audit-20261111-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
date -u
hostname
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'CODE'
from pathlib import Path
from collections import Counter
import json,hashlib,os
b=Path('/mnt/local/_backup/deep-llms_th2');a=b/'wc69-preservation-20261111-a01'
v=json.loads((a/'complete.json').read_text());assert v['complete']
assert hashlib.sha256((a/'files.jsonl').read_bytes()).hexdigest()==v['manifest_sha256']
rows=[];total=Counter();links=[]
for d,ds,fs in os.walk(b,followlinks=False):
 for n in ds:
  if (Path(d)/n).is_symlink():links.append(str(Path(d)/n))
 for n in fs:
  p=Path(d)/n
  if p.is_symlink():links.append(str(p));continue
  sz=p.stat().st_size;total['files']+=1;total['bytes']+=sz
  if sz<5<<20:continue
  if n.endswith(('.safetensors','.pt')) or n.startswith(('pytorch_model','model-')):continue
  rows.append(dict(path=str(p.relative_to(b)),bytes=sz))
print('BACKUP_AUDIT_TOTAL',json.dumps(dict(total)),flush=True)
print('LINKS',json.dumps(links),flush=True)
for row in sorted(rows,key=lambda v:v['path']):print('NONWEIGHT_LARGE',json.dumps(row),flush=True)
print('BACKUP_INPUT_AUDIT_COMPLETE',len(rows),flush=True)
CODE
