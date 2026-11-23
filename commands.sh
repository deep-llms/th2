#1 +20+a
#th2-ohmv-moe-dyn-failure-health-20261123-a01
set -u
date -u
/usr/bin/python3 - <<'PY'
# Read-only: failure reason, terminal/native receipt, per-job qualification status and log tails.
from pathlib import Path
import hashlib, json
base = Path('/mnt/local/_outputs/deep-llms_th2/moe_dyn_20261123_a01')
def j(p):
    return json.loads(p.read_text()) if p.is_file() else None
r = {n: j(base/n) for n in ('status.json', 'failure.json', 'terminal.json')}
nv = base/'native-verified.json'
r['native_verified_sha256'] = hashlib.sha256(nv.read_bytes()).hexdigest() if nv.is_file() else None
r['native_workers'] = j(nv)['samples'][-1]['workers'] if nv.is_file() else None
r['passed'] = sorted(str(p.relative_to(base)) for p in base.glob('loop/s*/qualification-*/passed.json'))
r['progress'] = {str(p.parent.relative_to(base)): j(p).get('step') for p in base.glob('**/progress.json')}
r['log_tails'] = {p.name: p.read_text()[-2500:] for p in sorted(base.glob('train-*.log')) if 'loop_dynattn' in p.name or 'Error' in p.read_text()[-3000:]}
r['queue_log_tail'] = (base/'execution/queue.log').read_text()[-1500:] if (base/'execution/queue.log').is_file() else None
print('MOE_DYN_FAIL_HEALTH', json.dumps(r), flush=True)
PY
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
