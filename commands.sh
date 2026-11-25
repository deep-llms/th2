#1 +20+a
#th2-ohmv-dot-f13-health-20261126-a01
set -u
date -u
/usr/bin/python3 - <<'PY'
# Read-only: f13 progress, finished NLLs (depth sweep, halting), downstream, terminal and receipt.
from pathlib import Path
import hashlib, json
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_vs_loops_2x_20261126_f13')
def j(p):
    return json.loads(p.read_text()) if p.is_file() else None
r = {n: j(base/n) for n in ('status.json', 'failure.json', 'terminal.json')}
nv = base/'native-verified.json'
r['native_verified_sha256'] = hashlib.sha256(nv.read_bytes()).hexdigest() if nv.is_file() else None
jobs = [(0, 'loop_cycle_a8_fa_2x'), (1, 'loop_cycle_a8_fa_2x'), (0, 'loop_cycle_a8_lr14_fa_2x'), (1, 'loop_cycle_a8_lr14_fa_2x'),
        (0, 'loop_prc_a8_fa_2x'), (1, 'loop_prc_a8_fa_2x'), (0, 'dense4_fa_2x'), (1, 'dense4_fa_2x')]
r['jobs'] = {}
for s, a in jobs:
    d = base/f's{s}'/a
    pr, sm = j(d/'progress.json'), j(d/'summary.json')
    ds = j(base/'downstream'/f's{s}-{a}'/'summary.json')
    r['jobs'][f's{s}/{a}'] = dict(step=(pr or {}).get('step'), tok_s=(pr or {}).get('tokens_per_s'), nll=(sm or {}).get('nll'),
                                  halting=(sm or {}).get('halting'),
                                  downstream={t: round(v['accuracy'], 4) for t, v in ds['tasks'].items()} if ds else None)
if r['failure.json']:
    r['log_tails'] = {p.name: p.read_text()[-2000:] for p in sorted(base.glob('*.log')) if 'Error' in p.read_text()[-4000:]}
print('F13_HEALTH', json.dumps(r), flush=True)
PY
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
