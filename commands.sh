#1 +20+a
#th2-wc69-family-f9-health-20261110-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.a2_family_f9_health --root /mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f9-20261110-a01
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 - <<'PY'
import json
from pathlib import Path
from scripts.gpu_process import process
root=Path('/mnt/local/_outputs/deep-llms_th2/wc69-a2-family-f9-20261110-a01')
print('F9_ROOT_EXISTS',root.exists())
for rel in ('source-manifest.json','launch.json','prepare-plan.json'):
 p=root/rel
 if p.is_file():
  v=json.loads(p.read_text());print('F9_LAUNCH_METADATA',rel,json.dumps({k:v[k] for k in ('code_commit','config','session') if k in v}))
for p in Path('/proc').glob('[0-9]*'):
 try:
  args=(p/'cmdline').read_bytes().split(b'\0')
  if any(x in args for x in (b'scripts.launch_a2_family_f9',b'scripts.a2_family_f9_prepare',b'scripts.a2_family_f9_run')):print('F9_KNOWN_PROCESS',json.dumps(process(int(p.name))))
 except (OSError,ValueError):pass
PY
