#1 +20+a
#th2-wc69-depth-p300-backup-20261108-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'BACKUP'
from pathlib import Path
import hashlib,json,tarfile
from datetime import datetime,timezone
root=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-depth-partc-20261103-a01/results/a2_depth_20261031')
output=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-depth-p300-backup-20261108-a01')
output.mkdir(exist_ok=False)
locks={'heals/ST1-dclm-P300-S1-12.8B-lr1e-3/result.json': '8cc3ed270ff861292681f04c65cb78ad0190d529db3117bc6e7736fefdce7a9a'}
spec='2cbef31e9bb6576ec9f3a20918ca69d7019e6e8e69e13bfc12bf92d151d2adbe'
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
  marker=json.loads((p.parent/'arm-complete.json').read_text())
  if marker['identity']!=result['full_state_identity'] or len(result['lr_updates'])!=48828:raise ValueError('Incomplete arm identity/schedule')
  for item in marker['files']:
   q=Path(item['path']);q.relative_to(p.parent)
   if 'final' not in q.relative_to(p.parent).parts and sha(q)!=item['sha256']:raise ValueError('Sealed science member changed')
 for q in p.parent.rglob('*'):
  relative=q.relative_to(p.parent)
  if relative.parts[0] in ('trainer','final') or not q.is_file():continue
  if q.is_symlink():raise ValueError('Unexpected symlink')
  if q.suffix not in ('.json','.jsonl','.npy','.csv','.md','.log','.yaml'):continue
  name=str(q.relative_to(root));files[name]=dict(bytes=q.stat().st_size,sha256=sha(q))
for name,expected in {'config/plan.json': '798208e92b5e31535eb1859c28c4ecbc83bdd2874438f7a8aa2148e8d744b7b4', 'DT6.json': '67993d6ef29f843d91b552a259e4a481e8a7113e82b3d7e39b10c6bc318dff19', 'DT6_checks.json': 'ffecae709dafbe753f9058682b944676bc32b0ef963763ebf04e291a68ee15bc'}.items():
 p=root/name
 if sha(p)!=expected:raise ValueError('Sealed file changed: '+name)
 files[name]=dict(bytes=p.stat().st_size,sha256=sha(p))
for q in [*(root/'perseq'/'ST1-dclm-P300-S1-12.8B-lr1e-3').glob('*.json'),root/'notes'/'P300.json',*sorted((root/'logs').glob('C-P300-attempt*.log'))]:
 if not q.is_file() or q.is_symlink():raise ValueError('Missing science file')
 files[str(q.relative_to(root))]=dict(bytes=q.stat().st_size,sha256=sha(q))
if len(list((root/'perseq'/'ST1-dclm-P300-S1-12.8B-lr1e-3').glob('*.json')))!=3:raise ValueError('Missing saved arrays')
archive=output/'depth_p300_science.tar.gz'
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
manifest=dict(spec_sha256=spec,utc=datetime.now(timezone.utc).isoformat(),scope='Completed Depth P300 12.8B with embedded and separate baseline/pre/post arrays, actual order, full LR, rank audits, resume history, logs, DT6 and config; excludes active P1T and weights. Weights in arm marker not independently rehashed by this backup. Not final Depth report',result_locks=locks,files=files,archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive)),members=len(files),raw_rank_traces=sum('sample-order-rank' in x for x in files),arrays=sum(x.endswith('.npy') for x in files),verified=True)
(output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('DEPTH_P300_BACKUP',json.dumps({k:x for k,x in manifest.items() if k not in ('files','result_locks')}),flush=True)
BACKUP
