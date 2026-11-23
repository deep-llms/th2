#1 +20+a
#th2-ohmv-sandwich-health-20261123-a01
set -u
date -u
/usr/bin/python3 - <<'PY'
# Read-only: sandwich progress per job, summaries when present, terminal/native receipt, export manifest.
from pathlib import Path
import hashlib, json
base = Path('/mnt/local/_outputs/deep-llms_th2/sandwich_20261123_a01')
def j(p):
    return json.loads(p.read_text()) if p.is_file() else None
r = {n: j(base/n) for n in ('status.json', 'failure.json', 'terminal.json')}
r['export_manifest'] = {k: v for k, v in (j(base/'export/manifest.json') or {}).items() if k != 'members'}
nv = base/'native-verified.json'
r['native_verified_sha256'] = hashlib.sha256(nv.read_bytes()).hexdigest() if nv.is_file() else None
r['native_workers'] = j(nv)['samples'][-1]['workers'] if nv.is_file() else None
r['jobs'] = {}
for name in ('sw_warm', 'not_warm', 'sw_m1', 'sw_m2', 'not_m1', 'not_m2', 'sw_stack', 'sw_stack_mix', 'not_stack', 'joint', 'dense5'):
    pr, sm = j(base/name/'progress.json'), j(base/name/'summary.json')
    r['jobs'][name] = dict(step=(pr or {}).get('step'), total=(pr or {}).get('total'), nll=(sm or {}).get('nll'))
if r['failure.json']:
    r['log_tails'] = {p.name: p.read_text()[-1500:] for p in sorted(base.glob('train-*.log')) if p.stat().st_size}
print('SANDWICH_HEALTH', json.dumps(r), flush=True)
PY
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
