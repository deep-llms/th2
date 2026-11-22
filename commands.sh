#1 +20+a
#th2-ohmv-loop-screen4-diagnose-20261123-a01
set -u
date -u
/usr/bin/python3 - <<'PY'
# Read-only: failure, terminal/native handoff and qualification log tails of the failed round-2 run.
from pathlib import Path
import hashlib, json
base = Path('/mnt/local/_outputs/deep-llms_th2/loop_screen4_20261123_a01')
def j(p):
    return json.loads(p.read_text()) if p.is_file() else None
r = {n: j(base/n) for n in ('failure.json', 'terminal.json', 'status.json')}
nv = base/'native-verified.json'
r['native_verified_sha256'] = hashlib.sha256(nv.read_bytes()).hexdigest() if nv.is_file() else None
r['native_workers'] = j(nv)['samples'][-1]['workers'] if nv.is_file() else None
r['passed'] = sorted(str(p.parent.relative_to(base)) for p in base.glob('s*/qualification-*/passed.json'))
print('R2_DIAG', json.dumps(r), flush=True)
for p in sorted(base.glob('qualify-*.log')):
    text = p.read_text()
    print('=== LOG', p.name, len(text), flush=True)
    print(text[-2500:] if text else '', flush=True)
PY
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
