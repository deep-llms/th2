#1 +20+a
#th2-6y3o-a2-data-reproduction-metadata-20261031-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
python3 -u - <<'PYCODE'
import json,gzip,base64,hashlib,socket
from pathlib import Path
assert socket.gethostname()=='thiennh-p6-6y3o-worker-0'
files={
'culturax_source_manifest.json':'/mnt/local/_data/deep-llms_th2/raw/cx_sampled_dd1b1513/subsets/qwen3_0.6b_base_en_30B/manifest.json',
'culturax_prepared_manifest.json':'/mnt/local/_outputs/deep-llms_th2/6y3o-a2-inputs-20261028-a01/run/inputs/data/manifest.json',
}
for name,path in files.items():
 p=Path(path);assert p.stat().st_size<10000000
 b=p.read_bytes();print('REPRO_FILE',json.dumps(dict(name=name,path=path,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),gzip_base64=base64.b64encode(gzip.compress(b,mtime=0)).decode())),flush=True)
print('REPRO_METADATA_COMPLETE',flush=True)
PYCODE
