#1 +20+a
#th2-ohmv-loop-stop2-health-20261123-a01
set -u
date -u
/usr/bin/python3 - <<'PY'
# Read-only: STOP round state, compact per-body summary (no per-row arrays), export manifest, native receipt.
from pathlib import Path
import hashlib, json
base = Path('/mnt/local/_outputs/deep-llms_th2/loop_stop2_20261123_a01')
def j(p):
    return json.loads(p.read_text()) if p.is_file() else None
r = {n: j(base/n) for n in ('status.json', 'failure.json', 'terminal.json')}
r['export_manifest'] = {k: v for k, v in (j(base/'export/manifest.json') or {}).items() if k != 'members'}
nv = base/'native-verified.json'
r['native_verified_sha256'] = hashlib.sha256(nv.read_bytes()).hexdigest() if nv.is_file() else None
r['native_workers'] = j(nv)['samples'][-1]['workers'] if nv.is_file() else None
bodies = {}
for p in sorted(base.glob('*/summary.json')):
    s = j(p)
    bodies[p.parent.name] = dict(fixed={k: [round(v['nll'], 5), round(v['passes'], 3)] for k, v in s['fixed'].items()},
                                 learned={k: [round(v['nll'], 5), round(v['passes'], 3), v['histogram']] for k, v in s['learned'].items()},
                                 random={k: [round(v['nll'], 5), round(v['passes'], 3)] for k, v in s['random_matched'].items()},
                                 oracle={k: [round(v['nll'], 5), round(v['passes'], 3)] for k, v in s['oracle_uniform_context'].items()},
                                 seconds=round(s['seconds']))
r['bodies'] = bodies
if r['failure.json']:
    r['log_tails'] = {p.name: p.read_text()[-1500:] for p in sorted(base.glob('stop-*.log'))}
print('LOOP_STOP_HEALTH', json.dumps(r), flush=True)
PY
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
