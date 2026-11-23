#1 +20+a
#th2-ohmv-dot-a04-a05-health-20261124-a01
set -u
date -u
/usr/bin/python3 - <<'PY'
# Read-only: a04 terminal/exports and a05 launch/progress; native receipts; GPU owners.
from pathlib import Path
import hashlib, json
out = Path('/mnt/local/_outputs/deep-llms_th2')
def j(p):
    return json.loads(p.read_text()) if p.is_file() else None
for run in ('dot_sandwich_20261124_a04', 'dot_sandwich_20261124_a05'):
    base = out/run
    r = {'exists': base.exists()}
    if base.exists():
        r.update({n: j(base/n) for n in ('status.json', 'failure.json', 'terminal.json', 'downstream_failure.json', 'rescore_failure.json')})
        r['exports'] = {p.name: {k: v for k, v in j(p).items() if k != 'members'} for p in sorted((base/'export').glob('*manifest.json'))} if (base/'export').exists() else {}
        nv = base/'native-verified.json'
        r['native_verified_sha256'] = hashlib.sha256(nv.read_bytes()).hexdigest() if nv.is_file() else None
        r['progress'] = {str(p.parent.relative_to(base)): j(p).get('step') for p in sorted(base.glob('s*/*/progress.json'))}
        r['nll'] = {str(p.parent.relative_to(base)): (j(p).get('nll') or {}).get('native') for p in sorted(base.glob('s*/*/summary.json'))}
        r['routes'] = {str(p.parent.relative_to(base)): {k: (j(p).get('dot_routes') or {}).get(k) for k in ('distinct_orders', 'modal_share')} for p in sorted(base.glob('s*/*/summary.json'))}
        r['downstream_done'] = sorted(p.parent.name for p in base.glob('downstream/*/summary.json'))
        r['rescore_done'] = sorted(p.parent.name for p in base.glob('rescore/*/summary.json'))
    print('RUN', run, json.dumps(r), flush=True)
PY
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
