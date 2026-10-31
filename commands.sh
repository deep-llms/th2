#1 +30+a
#th2-6y3o-quant-qt3-audit-package-20261031-a02
set -euo pipefail
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PYCODE'
import base64,hashlib,json,tarfile,io
from pathlib import Path
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-a2-quant-20261031-a01')
out=root/'qt3_revision_20261031_a02'
summary=out/'summary.json'
files=sorted(out.glob('*.json'))
print('AUDIT_PROGRESS',json.dumps({'files':len(files),'summary_exists':summary.exists()}),flush=True)
if summary.exists():
 s=json.loads(summary.read_text());print('AUDIT_SUMMARY',json.dumps({k:v for k,v in s.items() if k!='models'}),flush=True)
 buf=io.BytesIO()
 with tarfile.open(fileobj=buf,mode='w:gz') as tf:
  for p in files:tf.add(p,arcname=p.name,recursive=False)
 raw=buf.getvalue()
 print('AUDIT_BUNDLE',json.dumps({'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'files':len(files),'base64':base64.b64encode(raw).decode()}),flush=True)
else:
 log=root/'qt3_revision_20261031_a02.log'
 print(log.read_text()[-40000:] if log.exists() else 'No log yet',flush=True)
PYCODE
