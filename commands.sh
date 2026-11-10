#1 +20+a
#th2-wc69-family-f9-v110-health-20261110-a04
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.a2_family_f9_health --root /mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f9-v110-20261110-a01
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 - <<'PY_NODE'
import json
from pathlib import Path
r=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f9-v19-preserve-20261110-a01')
p=r/'backup/verified.json'
if p.exists():
 d=json.loads(p.read_text());print('PRESERVE_VERIFIED '+json.dumps({k:v for k,v in d.items() if k!='files'}))
else: print('PRESERVE_PENDING '+json.dumps({'root_exists':r.exists(),'snapshot_exists':(r/'snapshot.json').exists(),'archive_bytes':(r/'backup/family_f9_scientific.tar.gz').stat().st_size if (r/'backup/family_f9_scientific.tar.gz').exists() else None}))
PY_NODE
ps -eo pid,ppid,etime,args | grep '[s]cripts.a2_family_f9_preserve' || true
