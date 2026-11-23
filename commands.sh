#1 +240+a
#th2-ohmv-wait-a02-then-dot-sandwich-a03-20261124-r3
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
date -u
# 0) Read-only: why a01's downstream failed.
tail -c 2500 /mnt/local/_outputs/deep-llms_th2/dot_sandwich_20261123_a01/score-s1-sw_route.log || true
# 1) No signal: wait for a02 to end on its own, then salvage the valid no-id arms (a01 s0, a02 s1) into a small export.
/usr/bin/python3 - <<'PY'
import hashlib, json, tarfile, time
from pathlib import Path
out = Path('/mnt/local/_outputs/deep-llms_th2')
base = out/'dot_sandwich_20261123_a02'
for i in range(960):  # up to 80 min
    if (base/'terminal.json').exists():
        break
    if i % 120 == 0:
        st = json.loads((base/'status.json').read_text()) if (base/'status.json').exists() else None
        print('WAITING', time.strftime('%H:%M:%S'), json.dumps(st), flush=True)
    time.sleep(5)
t = json.loads((base/'terminal.json').read_text())
print('TERMINAL', json.dumps(t), flush=True)
print('FAILURE', (base/'failure.json').read_text() if (base/'failure.json').exists() else 'none', flush=True)
assert t['native_verified'] is True, 'native not verified'
salvage = out/'dot_sandwich_salvage_20261124'
salvage.mkdir(exist_ok=True)
names = {}
for run, rel in (('dot_sandwich_20261123_a01', 's0/sw_route_noid'), ('dot_sandwich_20261123_a02', 's1/sw_route_noid')):
    for f in ('summary.json', 'scores.npz', 'calibration.jsonl'):
        p = out/run/rel/f
        if p.is_file():
            names[f'{run}/{rel}/{f}'] = p
for run in ('dot_sandwich_20261123_a01', 'dot_sandwich_20261123_a02'):
    for p in sorted((out/run).glob('s*/*/summary.json')):
        names.setdefault(f'{run}/{p.parent.relative_to(out/run)}/summary.json', p)  # all summaries (small)
archive = salvage/'noid_and_summaries.tar.gz'
with tarfile.open(archive, 'w:gz') as tf:
    for n, p in sorted(names.items()):
        tf.add(p, arcname=n)
manifest = dict(archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), bytes=archive.stat().st_size,
                members={n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in names.items()})
(salvage/'manifest.json').write_text(json.dumps(manifest))
print('SALVAGE', json.dumps({k: v for k, v in manifest.items() if k != 'members'}), len(names), flush=True)
PY
out=/mnt/local/_outputs/deep-llms_th2
if [ -e "$out/dot_sandwich_20261123_a03" ]; then
  echo "A03_ALREADY_LAUNCHED"; cat "$out/dot_sandwich_20261123_a03/status.json" 2>/dev/null || true
  exit 0
fi
receipt=$(sha256sum $out/dot_sandwich_20261123_a02/native-verified.json | cut -d' ' -f1)
echo "NATIVE_RECEIPT $receipt"
# 3) Launch DOT sandwich a03 (bounded recipient ids, train export before downstream).
pkg="$PWD/resources/dot_sandwich_a03_20261123.tar.xz"
echo "62f88f83c330a932e8c7873c5a238bffad41c77facff606254a96d46de75eced  $pkg" | sha256sum -c -
extra=$(mktemp -d $out/dot_sandwich_a03_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
export PYTHONPATH=$out/dot_sandwich_20261123_a02/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/dot_sandwich_20261123_a03 \
  --old $out/dot_sandwich_20261123_a02/execution --prior-sha256 5ec4dc7bef68f0829e35e193255127fbbda8ba7906b6527c245242e15a2d4d69 \
  --added efficient_llm/warm_order/dot_sandwich.py efficient_llm/warm_order/loop_screen_downstream.py scripts/loop_screen_study.py scripts/dot_sandwich_study.py \
  --replaced efficient_llm/warm_order/dot_sandwich.py efficient_llm/warm_order/loop_screen_downstream.py scripts/loop_screen_study.py scripts/dot_sandwich_study.py \
  --module scripts.dot_sandwich_study \
  --handoff $out/dot_sandwich_20261123_a02/native-verified.json \
  --handoff-sha256 "$receipt"
/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_sandwich_20261123_a03')
jobs = [(0, 'sw_route'), (1, 'sw_route'), (0, 'sw_fixed'), (1, 'sw_fixed'), (0, 'sw_random'), (1, 'sw_route_noid'), (0, 'sw_rl'), (0, 'sw_cf')]
paths = {f's{s}/{a}': base/f's{s}'/a/'progress.json' for s, a in jobs}
for _ in range(60):
    if (base/'failure.json').is_file():
        print('DOT_FAILURE', (base/'failure.json').read_text()[-3000:], flush=True)
        break
    prog = {a: json.loads(p.read_text()) for a, p in paths.items() if p.is_file()}
    if len(prog) == len(paths):
        print('DOT_PROGRESS', json.dumps(prog), flush=True)
        break
    time.sleep(5)
print('DOT_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
