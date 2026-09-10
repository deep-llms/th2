#1 +30+a
#th2-swt-split-verified-eval-export-20260910-1803
set -euo pipefail
date -u
cd /mnt/local/deep-llms_th2
python3 - <<'PY'
import hashlib, json, zipfile
from datetime import datetime, timezone
from pathlib import Path
dest=Path('/mnt/local/_outputs/deep-llms_th2/swt/exports/eval98_all_20260910_a02')
archive=dest/'eval_results.zip'
assert json.loads((dest/'export_verification.json').read_text())['success'] is True
assert not (dest/'export_manifest.json').exists()
files=[]
with zipfile.ZipFile(archive) as z:
    assert len(z.namelist())==len(set(z.namelist()))==496
    for name in z.namelist():
        assert name.endswith('.json') and not name.startswith('/') and '..' not in Path(name).parts
        data=z.read(name)
        files.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
parts=[]
digest=hashlib.sha256()
with archive.open('rb') as src:
    while True:
        data=src.read(20_000_000)
        if not data: break
        path=dest/f'eval_results.zip.part{len(parts):03d}'
        with path.open('xb') as dst: dst.write(data)
        digest.update(data)
        parts.append(dict(name=path.name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
manifest=dict(success=True,source_run='/mnt/local/_outputs/deep-llms_th2/swt/next_capacity_eval98_diag294_ft126_20260910_a02',
    checkpoints=98,result_files=490,exported_utc=datetime.now(timezone.utc).isoformat(),
    files=files,parts=parts,archive=dict(name=archive.name,bytes=archive.stat().st_size,sha256=digest.hexdigest()))
with (dest/'export_manifest.json').open('x') as f:
    json.dump(manifest,f,indent=2,allow_nan=False)
    f.write('\n')
print('VERIFIED_EXPORT',json.dumps({k:v for k,v in manifest.items() if k!='files'}))
print('MANIFEST_SHA256',hashlib.sha256((dest/'export_manifest.json').read_bytes()).hexdigest())
print('ALL_EVAL_RESULTS_EXPORT_READY')
PY
