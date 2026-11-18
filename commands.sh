#1 +60+a
#th2-ohmv-local-order-terminal-health-20261118-a03
set -euo pipefail
export CUDA_VISIBLE_DEVICES=''
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -m scripts.local_order_health
python3 - <<'PYCODE'
from pathlib import Path,PurePosixPath
import json,hashlib,tarfile
base=Path('/mnt/local/_outputs/deep-llms_th2/local_order_20261118')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
if (base/'terminal.json').exists():
 terminal=json.loads((base/'terminal.json').read_text());assert terminal['success'] and terminal['native_verified']
 proof=json.loads((base/'backup.json').read_text());assert len(proof['node_checkpoint_copies'])==40
 for name,want in proof['node_checkpoint_copies'].items():assert sha(Path(name))==want,name
 archive=base/'results.tar.gz';assert sha(archive)==proof['archive_sha256'];assert archive.stat().st_size==proof['bytes']<25000000
 seen=set()
 with tarfile.open(archive,'r:gz') as t:
  for m in t.getmembers():
   rel=PurePosixPath(m.name);assert m.isfile() and not rel.is_absolute() and '..' not in rel.parts and rel.suffix!='.pt' and m.name not in seen
   assert hashlib.sha256(t.extractfile(m).read()).hexdigest()==proof['files'][m.name];seen.add(m.name)
 assert seen==set(proof['files']);assert not Path('/mnt/local/_gpu_guard/DISABLED').exists()
 print('TERMINAL_BACKUP_PROOF',json.dumps(dict(terminal=terminal,checkpoint_copies_rehashed=40,archive_bytes=proof['bytes'],archive_sha256=proof['archive_sha256'],proof_sha256=sha(base/'backup.json'),member_hashes_verified=len(seen),guard_released=True)))
else:print('TERMINAL_NOT_YET')
PYCODE
