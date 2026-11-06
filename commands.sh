#1 +20+a
#th2-wc69-a2-why-full-package-20261106-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PACKAGE'
from pathlib import Path
import json,tarfile,hashlib
from datetime import datetime,timezone
base=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-why-rest-20261105-a01')
root=base/'results/a2_why_20261104'
out=base/'exports/why_full_20261106_a01'
spec='87af7498eaffab01c2ab16e6e5093402a85ad55f292a8d182462e6543f2304ea'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
progress=json.loads((root/'progress.json').read_text())
decision=json.loads((root/'decision_why.json').read_text())
supervisor=json.loads((base/'supervised/supervisor.json').read_text())
assert progress['phase']=='Why_terminal' and decision['coverage_complete'] is True
assert decision['spec_sha256']==spec and supervisor['training_returncode']==0
names=[x['run_id'] for x in progress['completed']]
assert len(names)==len(set(names))==45 and all(x['returncode']==0 for x in progress['completed'])
for name in names:
 heal=root/'heals'/name
 r=json.loads((heal/'result.json').read_text())
 assert r['status']=='complete' and r['spec_sha256']==spec
 assert len(r['lr_updates'])==r['registered_run']['steps']
 assert (heal/'order.json').is_file() and len(list(heal.glob('tests-rank*.json')))==8
 assert len(list((root/'perseq'/name).glob('*.json')))==3
assert not out.exists();out.mkdir(parents=True)
files={}
excluded=[]
for p in sorted(root.rglob('*')):
 if not p.is_file():continue
 assert not p.is_symlink() and p.resolve().is_relative_to(root.resolve())
 if p.suffix in {'.json','.jsonl','.md','.log','.txt','.csv','.npy','.npz'}:
  files[str(p.relative_to(root))]=p
 else:excluded.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size))
for event in progress['completed']:
 name=event['run_id'];key='manifests/'+name+'.json'
 if key not in files:
  original=Path(event['result']['path']).parents[2]/key
  assert original.is_file() and not original.is_symlink()
  files[key]=original
for rel in ['source-manifest.json','supervised/supervisor.json','supervised/burn-verified.json','supervisor.log']:
 p=base/rel
 assert p.is_file() and not p.is_symlink()
 files['_provenance/'+rel]=p
entries=[dict(path=n,bytes=p.stat().st_size,sha256=sha(p)) for n,p in sorted(files.items())]
manifest=dict(spec_sha256=spec,created_at=datetime.now(timezone.utc).isoformat(),run_ids=names,files=entries,excluded=excluded,scope='Full Why scientific results, per-sequence arrays, actual sample orders, tests, logs, reports, manifests and runtime receipts; model weights/optimizer/training data excluded')
mp=out/'MANIFEST.json';mp.write_text(json.dumps(manifest,indent=2)+'\n')
archive=out/'why_full_20261106_a01.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for name,p in sorted(files.items()):tar.add(p,arcname=name,recursive=False)
 tar.add(mp,arcname='MANIFEST.json',recursive=False)
for e in entries:
 p=files[e['path']];assert p.stat().st_size==e['bytes'] and sha(p)==e['sha256']
parts=[]
with archive.open('rb') as f:
 i=0
 while chunk:=f.read(24*1024*1024):
  p=out/('why_full_20261106_a01.tar.gz.part%03d'%i)
  with p.open('xb') as w:w.write(chunk)
  parts.append(dict(path=str(p),bytes=len(chunk),sha256=sha(p)));i+=1
receipt=dict(spec_sha256=spec,archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),manifest_sha256=sha(mp),files=len(entries),runs=len(names),arrays=sum(n.startswith('perseq/') for n in files),parts=parts,receipt_path=str(out/'receipt.json'))
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('WHY_FULL_PACKAGE',json.dumps(receipt),flush=True)
PACKAGE
