#1 +20+a
#th2-wc69-family-f9-v110-transition-health-20261110-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.a2_family_f9_health --root /mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f9-20261110-a01
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 - <<'PY'
from pathlib import Path
import json
r=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f9-20261110-a01/scope_v110_drain')
print('DRAIN_RECEIPTS',json.dumps({p.name:json.loads(p.read_text()) for p in r.glob('*.json')}))
PY
