#1 +20+a
#th2-6y3o-a2-breadth-results-package-20261101-b01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PY'
from pathlib import Path
import json,hashlib,tarfile
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-a2-breadth-20261031-a01')
base=root/'results/a2_breadth_20261031'
out=root/'exports_final_b01';out.mkdir(exist_ok=False)
files=sorted(p for p in base.rglob('*') if p.is_file() and not p.is_symlink() and p.suffix in ('.json','.jsonl','.md') and 'final' not in p.relative_to(base).parts and 'trainer' not in p.relative_to(base).parts)
files += [root/x for x in ['source-manifest.json','jobs.json','budget.json','admission.json','supervised/supervisor.json','supervised/burn-verified.json','supervised/run/run.json','supervised/run/complete.json','supervised/run/breadth-complete.json']]
assert len(list((base/'perseq').glob('*/*.json')))==56
members=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
archive=out/'results.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for p in files:tar.add(p,arcname=str(p.relative_to(root)),recursive=False)
parts=[]
with archive.open('rb') as src:
 i=0
 while data:=src.read(20*1024*1024):
  p=out/f'part{i:03d}';p.write_bytes(data);parts.append(dict(path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()));i+=1
manifest=dict(root=str(root),files=members,archive_bytes=archive.stat().st_size,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),parts=parts)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('PACKAGE_MANIFEST',json.dumps(manifest),flush=True)
print('PACKAGE_COMPLETE',flush=True)
PY
