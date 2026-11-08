#1 +20+a
#th2-wc69-family-f6-base-backup-20261108-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'BACKUP'
from pathlib import Path
import hashlib,json,tarfile
from datetime import datetime,timezone
root=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f6-20261108-a02/results/a2_family_20261107')
output=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f6-base-backup-20261108-a01')
output.mkdir(exist_ok=False)
locks={'heals/ST1-fwe-S188-S1-0.8B-lr1e-3/result.json': '13205a0bd78cced0b6e1ab4970c0b1b7476748ec8083bf8948c220f96871a88b', 'heals/ST1-dclm-S72-S1-0.2B-lr1e-3/result.json': '6f6483d5ea955dc9167e118166d69a37795bc4f10442df69b3fcdca53268f65a', 'heals/ST1-dclm-S188-S1-0.8B-lr1e-3/result.json': 'd171297905ed137a8c0e885ed44af6494135478d758db0d0e24100461185010f', 'heals/ST1-fwe-S72-S1-0.8B-lr1e-3/result.json': 'f57f69e70adde3d2950788683374095601a3c81577338f21a4bb25f89c045dbc', 'heals/ST1-dclm-S72-S1-0.8B-lr1e-3/result.json': '5cd2fbe55e5fdde834680d2d212f79401e1a0208bb5b7aaca0a42a74b0f03095'}
spec='04233f076e8d82b6c76a33178d2944e621a278e01b779b3dce584b2b8ad464aa'
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
for name,expected in {'extension_decision.json': 'b9d7d1546e115ce59a7e42a841aa7e6b013564033e60cdd724db7a4b979c0fe9', 'config/plan.json': '8817b0131dafdcd6d864052dcf23b9083f7aebe2c5236f3e11d0bebdc4f0c72e'}.items():
 p=root/name
 if sha(p)!=expected:raise ValueError('Sealed file changed: '+name)
 files[name]=dict(bytes=p.stat().st_size,sha256=sha(p))
archive=output/'family_f6_base_science.tar.gz'
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
manifest=dict(spec_sha256=spec,utc=datetime.now(timezone.utc).isoformat(),scope='Completed FT9 and four F6 0.8B fits with all embedded baseline/pre/post arrays and raw rank traces; trigger/config locks; excludes active 3.2B fits and weights; not final v1.4 report',result_locks=locks,files=files,archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive)),members=len(files),raw_rank_traces=sum('sample-order-rank' in x for x in files),arrays=sum(x.endswith('.npy') for x in files),verified=True)
(output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('FAMILY_EARLY_BACKUP',json.dumps({k:x for k,x in manifest.items() if k not in ('files','result_locks')}),flush=True)
BACKUP
