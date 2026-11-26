#1 +20+a
#th2-ohmv-dot-f18-failure-read-20261126-a01
set -u
date -u
/usr/bin/python3 - <<'PY'
# Read-only: f18 failure, queue log tail, and the failing job's log tails; no writes, no process changes.
from pathlib import Path
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_f18_20261126_run')
f = base/'failure.json'
print('F18_FAILURE', f.read_text()[-1500:] if f.is_file() else 'none', flush=True)
q = base/'execution/queue.log'
print('QUEUE_TAIL', q.read_text()[-3000:] if q.is_file() else 'absent', flush=True)
for p in sorted(base.rglob('*.log'))[:40]:
    t = p.read_text(errors='replace')
    if 'Error' in t or 'Traceback' in t:
        print('LOG', str(p.relative_to(base)), t[-2500:], flush=True)
print('STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PY
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
