#1 +20+a
#th2-ohmv-dot-f01-f02-health-20261125-a01
set -u
date -u
/usr/bin/python3 - <<'PY'
# Read-only: f01 results (NLL, depth sweep, halting, downstream), f01 terminal and receipt; f02 launch state and progress.
from pathlib import Path
import hashlib, json
root = Path('/mnt/local/_outputs/deep-llms_th2')
def j(p):
    return json.loads(p.read_text()) if p.is_file() else None
def study(name, jobs):
    base = root/name
    r = {n: j(base/n) for n in ('status.json', 'failure.json', 'terminal.json')}
    nv = base/'native-verified.json'
    r['native_verified_sha256'] = hashlib.sha256(nv.read_bytes()).hexdigest() if nv.is_file() else None
    r['exists'] = base.is_dir()
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
    return r
f01 = [(0, 'swst_fixed_d3_film'), (1, 'swst_fixed_d3_film'), (0, 'swst_fixed_d3_lora'), (1, 'swst_fixed_d3_lora'),
       (0, 'swst_fixed_d4_light'), (1, 'swst_fixed_d4_light'), (0, 'swst_fixed_d3_fl'), (0, 'swst_fixed_d4_lightf')]
f02 = [(0, 'swst_d3_nt384'), (1, 'swst_d3_nt384'), (0, 'swst_m4d5_nt384'), (1, 'swst_m4d5_nt384'),
       (0, 'swst_d3_nt256'), (1, 'swst_d3_nt256'), (0, 'swst_m4d5_nt256'), (0, 'swst_d3_h4')]
print('F01_HEALTH', json.dumps(study('dot_fixed_improve_20261125_f01', f01)), flush=True)
print('F02_HEALTH', json.dumps(study('dot_compact_t_20261125_f02', f02)), flush=True)
PY
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
