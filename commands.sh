#1 +20+a
#th2-ohmv-moe-scale-health-20261123-a01
set -u
date -u
/usr/bin/python3 - <<'PY'
# Read-only: MoE-loop progress, summaries, terminal, export manifest, native receipt.
from pathlib import Path
import hashlib, json
base = Path('/mnt/local/_outputs/deep-llms_th2/moe_scale_20261123_a01')
def j(p):
    return json.loads(p.read_text()) if p.is_file() else None
r = {n: j(base/n) for n in ('status.json', 'failure.json', 'terminal.json')}
r['export_manifest'] = {k: v for k, v in (j(base/'export/manifest.json') or {}).items() if k != 'members'}
nv = base/'native-verified.json'
r['native_verified_sha256'] = hashlib.sha256(nv.read_bytes()).hexdigest() if nv.is_file() else None
r['native_workers'] = j(nv)['samples'][-1]['workers'] if nv.is_file() else None
jobs = [(0, 'attnres_moe_4x'), (0, 'loop_attnres_moe_4x'), (0, 'attnres8_moe_4x'), (1, 'attnres_moe_4x'),
        (1, 'loop_attnres_moe_4x'), (1, 'attnres_4x'), (1, 'loop_attnres_4x'), (1, 'attnres8_4x')]
r['jobs'] = {}
for s, a in jobs:
    d = base/f's{s}'/a
    pr, sm = j(d/'progress.json'), j(d/'summary.json')
    cal = (d/'calibration.jsonl').read_text().splitlines()[-1] if (d/'calibration.jsonl').is_file() else None
    r['jobs'][f's{s}/{a}'] = dict(step=(pr or {}).get('step'), last_cal=cal[:200] if cal else None,
                                  nll={k: v for k, v in ((sm or {}).get('nll') or {}).items() if k in ('native', 'native_p3', 'swap')},
                                  moe_trace={b: v.get('consecutive_overlap') for b, v in ((sm or {}).get('moe_trace') or {}).items()})
if r['failure.json']:
    r['log_tails'] = {p.name: p.read_text()[-1500:] for p in sorted(base.glob('train-*.log')) if 'Error' in p.read_text()[-3000:]}
print('MOE_SCALE_HEALTH', json.dumps(r), flush=True)
PY
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
