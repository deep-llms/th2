#1 +30+a
#th2-6y3o-efficient-kv-downstream-health-20261028-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/eval/bin/python3.11 -u - <<'READONLY'
import json,hashlib,socket
from pathlib import Path
from datetime import datetime,timezone
from scripts.gpu_process import inspect
assert socket.gethostname()=='thiennh-p6-6y3o-worker-0'
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-efficient-kv-downstream-20261028-a02')
print('HEALTH_AT',datetime.now(timezone.utc).isoformat(),flush=True)
print('GPU_INSPECTION',json.dumps(inspect()),flush=True)
for rel in ['jobs.json','source-manifest.json','input-identity.json','accelerate-env.log','supervised/supervisor.json','supervised/burn-verified.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/run/complete.json','supervised/run/run.json','supervised/run/adapter-check.json','supervised/run/smoke/summary.json','supervised/run/full/summary.json']:
 p=root/rel
 if p.is_file():
  raw=p.read_bytes()
  if len(raw)>2000000:raise ValueError('Unexpected oversized artifact '+rel)
  print('ARTIFACT',json.dumps(dict(path=rel,sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode())),flush=True)
 else:print('MISSING',rel,flush=True)
for p in sorted(root.rglob('*.log')):
 if p.is_file() and p.stat().st_size:
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-2500));tail=f.read().decode(errors='replace')
  print('LOG_TAIL',json.dumps(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,text=tail)),flush=True)
receipts=[]
for p in sorted((root/'supervised/run/full').glob('*/*.json')):
 with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
 receipts.append(dict(path=str(p.relative_to(root)),sha256=sha,bytes=p.stat().st_size))
print('FULL_RESULT_RECEIPTS',json.dumps(receipts),flush=True)
if (root/'supervised/run/full/summary.json').is_file():
 from eval.downstream_suite import GROUPS,PRIMARY
 counts={}
 for group,tasks in GROUPS.items():
  values={a:json.loads((root/'supervised/run/full'/a/(group+'.json')).read_text()) for a in ['A','B']}
  for task in tasks:
   metric=PRIMARY[task].split(',')[0]
   rows={a:{x['doc_id']:x for x in values[a]['samples'][task]} for a in values}
   assert rows['A'].keys()==rows['B'].keys()
   c=dict(both_wrong=0,A_only=0,B_only=0,both_correct=0)
   for doc_id in rows['A']:
    aa,bb=rows['A'][doc_id][metric],rows['B'][doc_id][metric]
    assert aa in [0,1] and bb in [0,1]
    c[['both_wrong','B_only','A_only','both_correct'][int(aa)*2+int(bb)]]+=1
   counts[task]=c
 print('PAIRED_ACCURACY_COUNTS',json.dumps(counts),flush=True)
print('HEALTH_COMPLETE',flush=True)
READONLY
