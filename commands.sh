#1 +20+a
#th2-wc69-backup-health-20261111-a04
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
date -u
ps -p 164676 -o pid,ppid,etime,pcpu,stat,args || true
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'CODE'
from pathlib import Path
import json,hashlib
r=Path('/mnt/local/_backup/deep-llms_th2/wc69-preservation-20261111-a01')
c=Path('/mnt/local/_outputs/deep-llms_th2/wc69-preservation-control-20261111-a01')
v=json.loads((r/'plan.json').read_text());v.pop('excluded',None);print('PLAN',json.dumps(v))
for p in (r/'progress.json',r/'complete.json',c/'exit.json'):
 print(str(p),p.read_text() if p.exists() else 'ABSENT')
log=c/'copy.log'
if log.exists():print('TAIL', '\n'.join(log.read_text().splitlines()[-12:]))
if (r/'complete.json').exists():
 v=json.loads((r/'complete.json').read_text());h=hashlib.sha256((r/'files.jsonl').read_bytes()).hexdigest();assert h==v['manifest_sha256'];print('MANIFEST_SHA_PASS',h)
CODE
