#1 +20+a
#th2-ohmv-dot-sandwich-a04-health-20261124-a01
set -u
date -u
/usr/bin/python3 - <<'PY'
# Read-only: DOT sandwich progress, finished NLLs and route statistics, downstream progress, terminal and receipt.
from pathlib import Path
import hashlib, json
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_sandwich_20261124_a04')
def j(p):
    return json.loads(p.read_text()) if p.is_file() else None
r = {n: j(base/n) for n in ('status.json', 'failure.json', 'terminal.json')}
r['export_manifest'] = {k: v for k, v in (j(base/'export/manifest.json') or {}).items() if k != 'members'}
nv = base/'native-verified.json'
r['native_verified_sha256'] = hashlib.sha256(nv.read_bytes()).hexdigest() if nv.is_file() else None
jobs = [(0, 'sw_prior'), (1, 'sw_prior'), (0, 'sw6_route'), (1, 'sw6_route'), (0, 'sw6_prior'), (1, 'sw6_prior'), (0, 'sw_cf_soft'), (0, 'sw_rl2')]
r['jobs'] = {}
for s, a in jobs:
    d = base/f's{s}'/a
    pr, sm = j(d/'progress.json'), j(d/'summary.json')
    routes = (sm or {}).get('dot_routes') or {}
    ds = j(base/'downstream'/f's{s}-{a}'/'summary.json')
    r['jobs'][f's{s}/{a}'] = dict(step=(pr or {}).get('step'), nll=(sm or {}).get('nll'),
                                  routes={k: routes.get(k) for k in ('distinct_orders', 'modal_share', 'usage')} if routes else None,
                                  downstream={t: round(v['accuracy'], 4) for t, v in ds['tasks'].items()} if ds else None)
if r['failure.json']:
    r['log_tails'] = {p.name: p.read_text()[-2000:] for p in sorted(base.glob('*.log')) if 'Error' in p.read_text()[-4000:]}
print('DOT_SW_HEALTH', json.dumps(r), flush=True)
PY
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
