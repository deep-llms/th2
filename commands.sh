#1 +20+a
#th2-wc69-family-f4-08-backup-20261108-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'BACKUP'
from pathlib import Path
import hashlib,json,tarfile
from datetime import datetime,timezone
root=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-20261107-a01/results/a2_family_20261107')
output=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f4-08-backup-20261108-a01')
output.mkdir(exist_ok=False)
locks={'reference/fwe/result.json': 'fd4b712e0abe0cafde74ab166a10f3c9d57f52cc951edc11908ea1b41e723d7c', 'reference/dclm/result.json': '20cab83ecb5c7e12a0f40665d2be7057da16b92c969d65bb4e9d89c40b8996d3', 'parents/k64/result.json': '27d2354578203a24e2e4f7f279a53b14dae4c62dc8f8847778f5d179d27f97f8', 'heals/F4-fwe-k64-BS1-0.8B-lr0.0001-s2030/result.json': 'f042ad581f4c50d310c8647873e5c93eddbafa8dccc42ede8fc89cc73b69d6ca', 'heals/F4-fwe-k64-BS1-0.2B-lr0.0001-s2030/result.json': 'f9adcd5f30364ba1d8b6c5748d15db8394d034b342b4c57e5585c0568555b1d1'}
spec='40089a98702a285d38799adcdfa89c72c7eec2e75e627cfef8552448f9a593c0'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
files={}
for rel,expected in locks.items():
 p=root/rel
 if sha(p)!=expected:raise ValueError('Completed result changed: '+rel)
 result=json.loads(p.read_text())
 if result['spec_sha256']!=spec:raise ValueError('Wrong spec')
 if rel.startswith('heals/'):
  if result['status']!='complete':raise ValueError('Unfinished run')
  for trace in result['raw_order_files']:
   if sha(trace['path'])!=trace['sha256']:raise ValueError('Raw sample trace changed')
 for q in p.parent.rglob('*'):
  relative=q.relative_to(p.parent)
  if relative.parts[0] in ('trainer','final') or not q.is_file():continue
  if q.is_symlink():raise ValueError('Unexpected symlink')
  if q.suffix not in ('.json','.jsonl','.npy','.csv','.md','.log','.yaml'):continue
  name=str(q.relative_to(root));files[name]=dict(bytes=q.stat().st_size,sha256=sha(q))
p=root/'lr_selection.json'
if sha(p)!='35dc83ca93eaa9d7d78ed98dfc351e2b275694c3085dbf3882b2fcb833870312':raise ValueError('LR selection changed')
files['lr_selection.json']=dict(bytes=p.stat().st_size,sha256=sha(p))
archive=output/'family_f4_08_science.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for rel in sorted(files):tar.add(root/rel,arcname=rel,recursive=False)
observed=set()
with tarfile.open(archive,'r:gz') as tar:
 for member in tar:
  if not member.isfile() or member.name not in files or member.name in observed:raise ValueError('Unexpected member')
  observed.add(member.name);data=tar.extractfile(member).read()
  if len(data)!=files[member.name]['bytes'] or hashlib.sha256(data).hexdigest()!=files[member.name]['sha256']:raise ValueError('Archive mismatch')
if observed!=set(files):raise ValueError('Missing member')
if archive.stat().st_size>=25*1024*1024:raise ValueError('Archive exceeds export cap')
manifest=dict(spec_sha256=spec,utc=datetime.now(timezone.utc).isoformat(),scope='Incremental completed F4 k64 0.2B and 0.8B plus k64 parent and reference baselines; earlier stages secured in prior verified exports; excludes active fits, weights, optimizer and data; not final Family report',result_locks=locks,files=files,archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive)),members=len(files),raw_rank_traces=sum('sample-order-rank' in x for x in files),arrays=sum(x.endswith('.npy') for x in files),verified=True)
(output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('FAMILY_EARLY_BACKUP',json.dumps({k:x for k,x in manifest.items() if k not in ('files','result_locks')}),flush=True)
BACKUP
