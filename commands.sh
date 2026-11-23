#1 +20+a
#th2-ohmv-moe-dyn-health-20261123-a01
set -u
date -u
/usr/bin/python3 - <<'PY'
# Read-only: moe/dyn a02 progress, summaries, terminal, export manifest, native receipt.
from pathlib import Path
import hashlib, json
base = Path('/mnt/local/_outputs/deep-llms_th2/moe_dyn_20261123_a02')
def j(p):
    return json.loads(p.read_text()) if p.is_file() else None
r = {n: j(base/n) for n in ('status.json', 'failure.json', 'terminal.json')}
r['export_manifest'] = {k: v for k, v in (j(base/'export/manifest.json') or {}).items() if k != 'members'}
nv = base/'native-verified.json'
r['native_verified_sha256'] = hashlib.sha256(nv.read_bytes()).hexdigest() if nv.is_file() else None
r['native_workers'] = j(nv)['samples'][-1]['workers'] if nv.is_file() else None
dirs = {a: base/'moe'/a for a in ('moe2_top1', 'moe4_top1', 'moe8_top2')}
dirs.update({f's{s}/{a}': base/'loop'/f's{s}'/a for s, a in ((0, 'dynattn'), (0, 'dynattn8'), (0, 'loop_dynattn'), (1, 'dynattn'), (1, 'loop_dynattn'))})
r['jobs'] = {}
for a, d in dirs.items():
    pr, sm = j(d/'progress.json'), j(d/'summary.json')
    cal = (d/'calibration.jsonl').read_text().splitlines()[-1] if (d/'calibration.jsonl').is_file() else None
    r['jobs'][a] = dict(step=(pr or {}).get('step'), last_cal=cal[:300] if cal else None,
                        nll={k: v for k, v in ((sm or {}).get('nll') or {}).items() if 'measure' in k or k in ('native', 'native_p3', 'swap')},
                        dyn_b=(sm or {}).get('attnres_dyn_b_norm'), usage=(sm or {}).get('usage'))
if r['failure.json']:
    r['log_tails'] = {p.name: p.read_text()[-1500:] for p in sorted(base.glob('train-*.log')) if 'Error' in p.read_text()[-3000:]}
print('MOE_DYN_HEALTH', json.dumps(r), flush=True)
PY
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
