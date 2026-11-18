#1 +30+a
#th2-ohmv-warm-order-result-only-package-20261118-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
python3 - <<'PYCODE'
from pathlib import Path,PurePosixPath
import json,hashlib,tarfile,os
base=Path('/mnt/local/_outputs/deep-llms_th2/warm_order_20261118')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
terminal=json.loads((base/'terminal.json').read_text());assert terminal['success'] and terminal['native_verified']
proof=json.loads((base/'backup.json').read_text())
assert sha(base/'execution/source.json')=='c2c3b055c91fe5174eea032e164f74a44d10f2da3d8cf66f4efd5cd1cf4eafc0'
for name,want in proof['node_checkpoint_copies'].items():assert sha(Path(name))==want,name
assert not Path('/mnt/local/_gpu_guard/DISABLED').exists()
out=base/'export-results-a01';out.mkdir(exist_ok=False)
files={name:want for name,want in proof['files'].items() if not name.endswith('.pt')}
for name in ['terminal.json','native-verified.json','backup.json']:
 files[name]=sha(base/name)
archive=out/'results.tar.gz'
with tarfile.open(archive,'w:gz') as t:
 for name,want in files.items():
  rel=PurePosixPath(name);assert not rel.is_absolute() and '..' not in rel.parts
  p=base/name;assert p.is_file() and not p.is_symlink() and sha(p)==want,name
  t.add(p,arcname=name,recursive=False)
assert archive.stat().st_size<25000000
result=dict(bytes=archive.stat().st_size,archive_sha256=sha(archive),files=files,node_checkpoint_copies=proof['node_checkpoint_copies'],excluded_weights=[n for n in proof['files'] if n.endswith('.pt')],node_backup_rehashed=True)
(out/'proof.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print('RESULTS_PACKAGE_READY',json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),proof=str(out/'proof.json'),proof_sha256=sha(out/'proof.json'),members=len(files),excluded_weights=result['excluded_weights'],backup_actual_rehash_pass=True)))
PYCODE
