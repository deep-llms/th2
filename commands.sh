#1 +25+a
#th2-ohmv-loop-screen-health-20261123-a01
set -u
date -u
/usr/bin/python3 - <<'PY'
# Read-only progress/terminal check. Packs a compact result archive ONLY after
# the supervisor wrote terminal.json with success (training and handoff finished).
from pathlib import Path
import hashlib, io, json, tarfile
base = Path('/mnt/local/_outputs/deep-llms_th2/loop_screen_20261123_a01')
arms = ('dense4','dense4_2x','dense8','serial','memory','gated','gated_var','attnres')
def j(p):
    return json.loads(p.read_text()) if p.is_file() else None
r = {n: j(base/n) for n in ('status.json', 'failure.json', 'terminal.json')}
r['report_present'] = (base/'report.json').is_file()
r['arms'] = {}
for a in arms:
    d = base/'s0'/a
    s = j(d/'summary.json')
    r['arms'][a] = dict(progress=j(d/'progress.json'),
                        nll=(s or {}).get('nll'), train_seconds=(s or {}).get('train_seconds'))
nv = j(base/'native-verified.json')
r['native_workers'] = nv['samples'][-1]['workers'] if nv else None
r['native_verified_sha256'] = hashlib.sha256((base/'native-verified.json').read_bytes()).hexdigest() if nv else None
for n, p in (('requests', Path('/mnt/local/_outputs/deep-llms_th2/warm_downstream_20261118_a02/requests.json')),
             ('raw_template', Path('/mnt/local/_outputs/deep-llms_th2/warm_downstream_replay_20261118_a01/runs/pilot-D-s0/raw.json'))):
    r[n] = dict(exists=p.is_file(), bytes=p.stat().st_size if p.is_file() else None)
print('LOOP_SCREEN_HEALTH', json.dumps(r), flush=True)
term = r['terminal.json']
if term and term.get('success') and term.get('native_verified') and r['report_present']:
    out = base/'export'
    out.mkdir(exist_ok=True)
    names = ['report.json', 'backup.json', 'terminal.json', 'native-verified.json', 'prepared.json',
             'execution/source.json']
    for a in arms:
        names += [f's0/{a}/{n}' for n in ('summary.json', 'scores.npz', 'calibration.jsonl', 'terminal.json')]
    names += [f's0/qualification-{a}/passed.json' for a in arms]
    archive = out/'loop_screen_results.tar.gz'
    if not archive.exists():
        with tarfile.open(archive, 'w:gz') as t:
            for n in names:
                t.add(base/n, arcname=n)
    members = {n: hashlib.sha256((base/n).read_bytes()).hexdigest() for n in names}
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(dict(archive_sha256=digest, bytes=archive.stat().st_size,
                                                     members=members), indent=1))
    print('LOOP_SCREEN_ARCHIVE', digest, archive.stat().st_size, len(names), flush=True)
PY
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
