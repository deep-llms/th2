#1 +20+a
#th2-wc69-backup-prune-execute-20261111-a01
set -euo pipefail
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
hostname
date -u
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u scripts/a2_prune_backup_inputs.py execute --plan-sha e929484f849c95bad95797af484e78fa456882ae003d8a668f291da644d37412
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 - <<'PY'
from pathlib import Path
import hashlib,json
b=Path('/mnt/local/_backup/deep-llms_th2');a=b/'wc69-input-prune-20261111-a01'
assert json.loads((a/'complete.json').read_text())['complete']
for src in ['docs/BACKUP_RETENTION_20261111.md','scripts/a2_prune_backup_inputs.py']:
 p=Path(src);dst=a/p.name
 with dst.open('xb') as f:f.write(p.read_bytes())
 assert hashlib.sha256(dst.read_bytes()).digest()==hashlib.sha256(p.read_bytes()).digest()
files=[p for p in b.rglob('*') if p.is_file()]
print('BACKUP_FINAL_TOTAL',json.dumps(dict(files=len(files),bytes=sum(p.stat().st_size for p in files))))
print('BACKUP_PRUNE_HANDOFF_COMPLETE')
PY
