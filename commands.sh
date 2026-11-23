#1 +20+a
#th2-ohmv-router-health-20261123-a01
set -u
date -u
/usr/bin/python3 - <<'PY'
# Read-only: router screen progress, summaries (NLL, choice fractions, usage), terminal, export, native receipt.
from pathlib import Path
import hashlib, json
base = Path('/mnt/local/_outputs/deep-llms_th2/router_screen_20261123_a01')
def j(p):
    return json.loads(p.read_text()) if p.is_file() else None
r = {n: j(base/n) for n in ('status.json', 'failure.json', 'terminal.json')}
r['export_manifest'] = {k: v for k, v in (j(base/'export/manifest.json') or {}).items() if k != 'members'}
nv = base/'native-verified.json'
r['native_verified_sha256'] = hashlib.sha256(nv.read_bytes()).hexdigest() if nv.is_file() else None
r['native_workers'] = j(nv)['samples'][-1]['workers'] if nv.is_file() else None
r['arms'] = {}
for a in ('single', 'seq_switch2', 'seq_switch3', 'seq_cf2', 'order_cf', 'both_mix', 'tok_moe2', 'tok_moe3'):
    pr, sm = j(base/a/'progress.json'), j(base/a/'summary.json')
    cal = (base/a/'calibration.jsonl').read_text().splitlines()[-1] if (base/a/'calibration.jsonl').is_file() else None
    r['arms'][a] = dict(step=(pr or {}).get('step'), usage=(pr or {}).get('usage'), last_cal=cal,
                        nll=(sm or {}).get('nll'), choice=(sm or {}).get('choice'))
if r['failure.json']:
    r['log_tails'] = {p.name: p.read_text()[-1500:] for p in sorted(base.glob('train-*.log')) if p.stat().st_size}
print('ROUTER_HEALTH', json.dumps(r), flush=True)
PY
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
