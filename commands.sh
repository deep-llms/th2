#1 +20+a
#th2-ohmv-liveness-f21-state-20261126-a01
set -u
date -u
/usr/bin/python3 - <<'PY'
# Read-only liveness probe (a01): did the f21 run start (its directory), its status and progress; no writes, no process changes.
from pathlib import Path
import json
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_f21_20261126')
print('F21_EXISTS', base.is_dir(), flush=True)
for n in ('status.json', 'failure.json', 'terminal.json'):
    p = base/n
    print(n, p.read_text()[-600:] if p.is_file() else 'absent', flush=True)
for p in sorted(base.glob('s*/*/progress.json')):
    print('PROGRESS', p.parent.parent.name + '/' + p.parent.name, p.read_text()[:200], flush=True)
PY
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader | head -8
