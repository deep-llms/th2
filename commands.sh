#1 +30+a
#th2-6y3o-a2-mech-token-backup-package-20261031-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PY'
import hashlib,json,socket
from pathlib import Path
from datetime import datetime,timezone
assert socket.gethostname()=='thiennh-p6-6y3o-worker-0'
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-a2-mech-20261031-a01')
base=root/'results/a2_mech_20261031';out=root/'exports/token_backup_a01';out.mkdir(parents=True,exist_ok=False)
inv=json.loads((base/'readout_frequency_revision/token_inventory.json').read_text());assert len(inv['files'])==18
receipt=dict(spec_sha256=inv['spec_sha256'],created_at=datetime.now(timezone.utc).isoformat(),no_gpu_actions=True,files=[])
for i,row in enumerate(inv['files']):
 src=Path(row['path']);assert src.resolve().is_relative_to((base/'workers/tokens').resolve())
 folder=out/f'b{i//6+1:02d}';folder.mkdir(exist_ok=True)
 pieces=[];h=hashlib.sha256()
 with src.open('rb') as stream:
  part=0
  while block:=stream.read(20*1024*1024):
   h.update(block);p=folder/f'array{i:02d}.part{part:02d}';p.write_bytes(block)
   pieces.append(dict(path=str(p),bytes=len(block),sha256=hashlib.sha256(block).hexdigest()));part+=1
 assert h.hexdigest()==row['sha256'],str(src)
 receipt['files'].append(dict(index=i,relative_path=str(src.relative_to(base/'workers/tokens')),original=str(src),sha256=h.hexdigest(),bytes=src.stat().st_size,parts=pieces))
(out/'manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('TOKEN_BACKUP_PACKAGE',json.dumps(receipt),flush=True)
PY
