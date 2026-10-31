#1 +30+a
#th2-6y3o-a2-mech-package-20261031-b01
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PY'
import hashlib,json,tarfile,socket
from pathlib import Path
from datetime import datetime,timezone
assert socket.gethostname()=='thiennh-p6-6y3o-worker-0'
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-a2-mech-20261031-a01')
base=root/'results/a2_mech_20261031'
complete=json.loads((root/'cpu_completion/complete.json').read_text())
assert complete['readout_returncode']==0
out=root/'exports/final_b01';out.mkdir(parents=True,exist_ok=False)
files=set(p for p in base.rglob('*') if p.is_file() and p.suffix in ('.json','.md') and not p.is_symlink())
for rel in ['source-manifest.json','budget.json','admission.json','jobs.json','supervised/supervisor.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/burn-verified.json','supervised/run/run.json','cpu_completion/admission.json','cpu_completion/complete.json','cpu_completion/source-manifest.json']:
 p=root/rel
 if p.exists():files.add(p)
manifest={'created_at':datetime.now(timezone.utc).isoformat(),'spec_sha256':'82093a3fa9ae6d9428c007e8d26c112d875dae0b84dfe83eee3677cf4c8184d4','files':{},'token_arrays_retained_on_node':[]}
for p in sorted(files):
 manifest['files'][str(p.relative_to(root))]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size,'mtime_utc':datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat()}
for p in sorted((base/'workers/tokens').glob('*/*/*.npy')):
 manifest['token_arrays_retained_on_node'].append({'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
mp=out/'manifest.json';mp.write_text(json.dumps(manifest,separators=(',',':'))+'\n')
tar=out/'results.tar.gz'
with tarfile.open(tar,'w:gz') as archive:
 for p in sorted(files):archive.add(p,arcname=str(p.relative_to(root)),recursive=False)
parts=[]
with tar.open('rb') as f:
 index=0
 while data:=f.read(16*1024*1024):
  p=out/f'results.tar.gz.part{index:03d}';p.write_bytes(data);parts.append({'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()});index+=1
receipt={'manifest':{'path':str(mp),'sha256':hashlib.sha256(mp.read_bytes()).hexdigest()},'archive_sha256':hashlib.sha256(tar.read_bytes()).hexdigest(),'file_count':len(files),'parts':parts,'token_arrays':len(manifest['token_arrays_retained_on_node'])}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('MECH_PACKAGE',json.dumps(receipt),flush=True)
PY
