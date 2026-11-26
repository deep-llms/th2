#1 +20+a
#th2-ohmv-liveness-f18b-state-20261126-a01
set -u
date -u
/usr/bin/python3 - <<'PY'
# Read-only liveness probe: does the f18b run directory exist yet, and its status (no writes, no process changes).
from pathlib import Path
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_f18b_20261126')
print('F18B_EXISTS', base.is_dir(), flush=True)
for n in ('status.json', 'failure.json', 'terminal.json'):
    p = base/n
    print(n, p.read_text()[-600:] if p.is_file() else 'absent', flush=True)
PY
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader | head -8
