#1 +20+a
#th2-wc69-family-v17-s72-backup-20261109-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'BACKUP'
from pathlib import Path
import hashlib,json,tarfile
from datetime import datetime,timezone
root=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-v17-20261109-a02/results/a2_family_20261107')
output=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-v17-s72-backup-20261109-a01')
output.mkdir(exist_ok=False)
locks={'heals/ST1-dclm-S72-S1-3.2B-lr1e-3/result.json': 'f87e41f5a501b637bce43d06f12abd7da67e1a5aa62b0140341078d4bfcc05d9'}
spec='70bee591aa19a48578998655b53404e358f218a2cbcb1946c432b00cd5f44209'
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
for name,expected in {'config/plan.json': '6cc0ad2c59b62fbe86e3a254f32883276bfa8249ba07f36c5f06711329317742'}.items():
 p=root/name
 if sha(p)!=expected:raise ValueError('Sealed file changed: '+name)
 files[name]=dict(bytes=p.stat().st_size,sha256=sha(p))
archive=output/'family_v17_s72_science.tar.gz'
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
manifest=dict(spec_sha256=spec,utc=datetime.now(timezone.utc).isoformat(),scope='Completed F7 DCLM S72 S1 3.2B with embedded baseline/pre/post arrays and eight raw rank traces; config lock; excludes active F7 S188 and weights; not final v1.7 report',result_locks=locks,files=files,archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive)),members=len(files),raw_rank_traces=sum('sample-order-rank' in x for x in files),arrays=sum(x.endswith('.npy') for x in files),verified=True)
(output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('FAMILY_EARLY_BACKUP',json.dumps({k:x for k,x in manifest.items() if k not in ('files','result_locks')}),flush=True)
BACKUP
