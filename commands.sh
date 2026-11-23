#1 +20+a
#th2-ohmv-family-downstream-health-20261123-a01
set -u
date -u
/usr/bin/python3 - <<'PY'
# Read-only: family downstream terminal, per-unit accuracies, export manifest and native receipt hash.
from pathlib import Path
import hashlib, json
base = Path('/mnt/local/_outputs/deep-llms_th2/family_downstream_20261123_a01')
def j(p):
    return json.loads(p.read_text()) if p.is_file() else None
r = {n: j(base/n) for n in ('status.json', 'failure.json', 'terminal.json')}
rep = j(base/'report.json') or {}
r['report'] = {u: {route: {t: round(v['accuracy'], 4) for t, v in tasks.items()} for route, tasks in routes.items()} for u, routes in rep.items()}
r['export_manifest'] = {k: v for k, v in (j(base/'export/manifest.json') or {}).items() if k != 'members'}
nv = base/'native-verified.json'
r['native_verified_sha256'] = hashlib.sha256(nv.read_bytes()).hexdigest() if nv.is_file() else None
r['native_workers'] = j(nv)['samples'][-1]['workers'] if nv.is_file() else None
r['units_done'] = len(list(base.glob('*/summary.json')))
if r['failure.json']:
    r['log_tails'] = {p.name: p.read_text()[-400:] for p in sorted(base.glob('score-*.log'))}
print('FAMILY_DS_HEALTH', json.dumps(r), flush=True)
PY
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
