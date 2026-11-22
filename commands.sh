#1 +20+a
#th2-ohmv-loop-screen2-downstream-health-20261123-a01
set -u
date -u
/usr/bin/python3 - <<'PY'
# Read-only: downstream scoring progress/terminal, export manifest and native receipt hash.
from pathlib import Path
import hashlib, json
base = Path('/mnt/local/_outputs/deep-llms_th2/loop_screen2_downstream_20261123_a01')
def j(p):
    return json.loads(p.read_text()) if p.is_file() else None
r = {n: j(base/n) for n in ('status.json', 'failure.json', 'terminal.json', 'report.json')}
r['export_manifest'] = {k: v for k, v in (j(base/'export/manifest.json') or {}).items() if k != 'members'}
nv = base/'native-verified.json'
r['native_verified_sha256'] = hashlib.sha256(nv.read_bytes()).hexdigest() if nv.is_file() else None
r['native_workers'] = j(nv)['samples'][-1]['workers'] if nv.is_file() else None
r['arms_done'] = sorted(p.parent.name for p in base.glob('*/summary.json'))
logs = sorted(base.glob('score-*.log'))
r['log_tails'] = {p.name: p.read_text()[-300:] for p in logs} if r['failure.json'] else {}
print('DOWNSTREAM_HEALTH', json.dumps(r), flush=True)
PY
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
