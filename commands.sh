#1 +20+a
#th2-ohmv-scale-health-20261123-a02
set -u
date -u
/usr/bin/python3 - <<'PY'
# Read-only: scale round progress per job, summaries when present, terminal/native receipt, export manifest.
from pathlib import Path
import hashlib, json
base = Path('/mnt/local/_outputs/deep-llms_th2/scale_round_20261123_a01')
def j(p):
    return json.loads(p.read_text()) if p.is_file() else None
r = {n: j(base/n) for n in ('status.json', 'failure.json', 'terminal.json')}
r['export_manifest'] = {k: v for k, v in (j(base/'export/manifest.json') or {}).items() if k != 'members'}
nv = base/'native-verified.json'
r['native_verified_sha256'] = hashlib.sha256(nv.read_bytes()).hexdigest() if nv.is_file() else None
r['native_workers'] = j(nv)['samples'][-1]['workers'] if nv.is_file() else None
jobs = {a: base/'loop'/'s0'/a for a in ('loop_attnres_4x', 'attnres_4x', 'attnres8_4x', 'dense8_4x', 'dense4_4x')}
jobs.update({'domain_'+a: base/'domain'/a for a in ('single', 'oracle')})
r['jobs'] = {}
for a, d in jobs.items():
    pr, sm = j(d/'progress.json'), j(d/'summary.json')
    cal = (d/'calibration.jsonl').read_text().splitlines()[-1] if (d/'calibration.jsonl').is_file() else None
    r['jobs'][a] = dict(step=(pr or {}).get('step'), total=(pr or {}).get('total'), last_cal=cal,
                        nll=(sm or {}).get('nll'), routing=(sm or {}).get('routing'))
if r['failure.json']:
    r['log_tails'] = {p.name: p.read_text()[-1500:] for p in sorted(base.glob('train-*.log')) if p.stat().st_size}
print('SCALE_HEALTH', json.dumps(r), flush=True)
PY
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
