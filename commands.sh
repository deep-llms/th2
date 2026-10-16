#1 +30+a
#th2-q359-ce7454-data-verify-20261016-a01
set -euo pipefail
date -u
hostname
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/eval_fa4/bin/python - <<'VERIFY'
import hashlib, json, zipfile
from pathlib import Path
root=Path('/mnt/local/_data/deep-llms_th2/ce7454/raw')
manifest=json.loads((root/'manifest.json').read_text())
with (root/'train.zip').open('rb') as f:
    sha=hashlib.file_digest(f,'sha256').hexdigest()
assert sha == manifest['sha256'] == 'ef57baacee12b92038a845308b0911dd71c5b564bdda43b516ed2f90fbb08ce3'
with zipfile.ZipFile(root/'train.zip') as z:
    rows=[r for r in z.infolist() if not r.is_dir()]
    assert len(rows)==15000
    assert {r.filename for r in rows} == {r['path'] for r in manifest['images']}
print('CE7454_REMOTE_ARCHIVE_VERIFIED',json.dumps({'sha256':sha,'images':len(rows),'bytes':(root/'train.zip').stat().st_size}))
VERIFY
