#1 +30+a
#th2-6y3o-a2-review-inventory-20261028-a01
set -euo pipefail
date -u
hostname
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
python3 scripts/gpu_status.py
python3 - <<'NODE'
from pathlib import Path
import json
for root in ('/mnt/local/conda-py311/envs','/mnt/local/uvenvs'):
    p=Path(root)
    print('ENV_DIRECTORIES',root,sorted(x.name for x in p.iterdir()) if p.exists() else [])
for name in ('/mnt/local/_gpu_guard/owner.json','/mnt/local/_gpu_guard/state.json'):
    p=Path(name)
    print('GUARD_RECORD',name,p.read_text() if p.is_file() else 'absent')
print('A2_ENV_EXISTS',Path('/mnt/local/conda-py311/envs/a2_env').exists())
NODE
