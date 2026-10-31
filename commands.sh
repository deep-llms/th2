#1 +30+a
#th2-6y3o-quant-results-package-20261031-b02
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
python3 -u - <<'PY'
"""Copy existing registered result artifacts into bounded retrieval bundles; no model work."""
import hashlib,json,os,sys,tarfile,time
from pathlib import Path
from datetime import datetime,timezone
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01')
base=root/'results/a2_quant_20261031'
out=root/'retrieval_20261031_b02'
out.mkdir(exist_ok=False)
sys.path.insert(0,str(root/'source'))
from scripts.gpu_process import inspect

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(name,obj):
 p=out/name;p.write_text(json.dumps(obj,indent=2)+'\n');return p
sm=json.loads((root/'source-manifest.json').read_text())
changed=[n for n,v in sm['files'].items() if sha(root/'source'/n)!=v]
source=write('source_verification.json',dict(code_commit=sm['code_commit'],count=len(sm['files']),changed=changed))
samples=[]
for i in range(2):
 snap=inspect();p=root/'supervised/burn.log'
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-5000));tail=f.read().decode(errors='replace')
 samples.append(dict(inspection=snap,burn_tail=tail,utc=datetime.now(timezone.utc).isoformat()))
 if i==0:time.sleep(12)
terminal=write('terminal_samples.json',samples)
core=[]
for p in sorted(base.rglob('*')):
 if not p.is_file() or p.suffix not in ('.json','.jsonl','.md'):continue
 rel=p.relative_to(base)
 if rel.parts[0] in ('downstream','replays','independent_replays'):continue
 core.append((p,str(p.relative_to(root))))
for p in sorted((base/'downstream').glob('*/*/tests.json')):core.append((p,str(p.relative_to(root))))
for rel in ('source-manifest.json','admission.json','budget.json','jobs.json','inspection.json','supervised/supervisor.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/burn-verified.json','supervised/run/run.json','supervised/run/complete.json','supervised/run/quant-complete.json'):
 p=root/rel
 if p.is_file():core.append((p,rel))
core.extend([(source,'verification/source_verification.json'),(terminal,'verification/terminal_samples.json')])
down=json.loads((base/'downstream.json').read_text())
groups={'core':core,'scores':[],'replays':[]}
for kind in ('scores','replays','independent_replays'):
 for label,item in sorted(down[kind].items()):
  p=Path(item['path'])
  if not p.resolve().is_relative_to(Path('/mnt/local/_outputs/deep-llms_th2')):raise ValueError('Unexpected result path')
  if sha(p)!=item['sha256']:raise ValueError('Locked artifact changed: '+label)
  ds,model=label.split('/')
  groups['scores' if kind=='scores' else 'replays'].append((p,kind+'/'+ds+'/'+model+'.json'))
manifest={'created_at':datetime.now(timezone.utc).isoformat(),'root':str(root),'spec_sha256':json.loads((base/'complete.json').read_text())['spec_sha256'],'groups':{},'source_changed':changed}
for name,files in groups.items():
 records=[];archive=out/(name+'.tar.gz')
 with tarfile.open(archive,'w:gz',compresslevel=3) as tf:
  for p,arc in files:
   h=sha(p);size=p.stat().st_size
   tf.add(p,arcname=arc,recursive=False)
   if sha(p)!=h:raise ValueError('Artifact changed during copy: '+str(p))
   records.append(dict(path=str(p),member=arc,bytes=size,sha256=h))
 parts=[]
 with archive.open('rb') as stream:
  i=0
  while True:
   chunk=stream.read(16*1024*1024)
   if not chunk:break
   part=out/(name+'.part'+str(i).zfill(3));part.write_bytes(chunk)
   parts.append(dict(path=str(part),bytes=len(chunk),sha256=sha(part)));i+=1
 manifest['groups'][name]=dict(files=records,archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,parts=parts)
 print('BUNDLE',json.dumps(dict(name=name,file_count=len(records),bytes=archive.stat().st_size,parts=parts)),flush=True)
p=write('manifest.json',manifest)
print('RETRIEVAL_MANIFEST',json.dumps(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))),flush=True)
print('PACKAGE_COMPLETE',flush=True)

PY
