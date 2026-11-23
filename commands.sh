#1 +25+a
#th2-ohmv-loop-screen4-health-20261123-a02
set -u
date -u
/usr/bin/python3 - <<'PY'
# Read-only progress/terminal check of round 2 a02; packs results only after verified terminal + handoff.
from pathlib import Path
import hashlib, json, tarfile
base = Path('/mnt/local/_outputs/deep-llms_th2/loop_screen4_20261123_a02')
jobs = [(0,'loop_attnres_tokvar'),(0,'gated_tokvar'),(1,'loop_attnres'),(1,'attnres'),(1,'dense4'),(1,'attnres8'),(0,'loop_attnres_2x'),(0,'dense8_2x')]
def j(p):
    return json.loads(p.read_text()) if p.is_file() else None
r = {n: j(base/n) for n in ('status.json', 'failure.json', 'terminal.json')}
r['report_present'] = (base/'report.json').is_file()
r['jobs'] = {}
for s, a in jobs:
    d = base/f's{s}'/a
    sm = j(d/'summary.json') or {}
    r['jobs'][f's{s}/{a}'] = dict(progress=j(d/'progress.json'), nll=sm.get('nll'))
nv = base/'native-verified.json'
r['native_verified_sha256'] = hashlib.sha256(nv.read_bytes()).hexdigest() if nv.is_file() else None
r['native_workers'] = j(nv)['samples'][-1]['workers'] if nv.is_file() else None
print('LOOP_SCREEN4_HEALTH', json.dumps(r), flush=True)
term = r['terminal.json']
if term and term.get('success') and term.get('native_verified') and r['report_present']:
    out = base/'export'
    out.mkdir(exist_ok=True)
    names = ['report.json', 'backup.json', 'terminal.json', 'native-verified.json', 'prepared.json', 'execution/source.json']
    for s, a in jobs:
        names += [f's{s}/{a}/{n}' for n in ('summary.json', 'scores.npz', 'calibration.jsonl', 'terminal.json')]
        names.append(f's{s}/qualification-{a}/passed.json')
    archive = out/'loop_screen4_results.tar.gz'
    if not archive.exists():
        with tarfile.open(archive, 'w:gz') as t:
            for n in names:
                t.add(base/n, arcname=n)
    members = {n: hashlib.sha256((base/n).read_bytes()).hexdigest() for n in names}
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(dict(archive_sha256=digest, bytes=archive.stat().st_size, members=members), indent=1))
    print('LOOP_SCREEN4_ARCHIVE', digest, archive.stat().st_size, len(names), flush=True)
PY
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
