#1 +240+a
#th2-ohmv-wait-a01-then-dot-sandwich-a02-20261123
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
date -u
# 1) No signal: wait for DOT sandwich a01 to end on its own (training, downstream, native restore + verification).
/usr/bin/python3 - <<'PY'
import json, time
from pathlib import Path
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_sandwich_20261123_a01')
for i in range(540):  # up to 45 min
    if (base/'terminal.json').exists():
        break
    if i % 60 == 0:
        st = json.loads((base/'status.json').read_text()) if (base/'status.json').exists() else None
        print('WAITING', time.strftime('%H:%M:%S'), json.dumps(st), flush=True)
    time.sleep(5)
t = json.loads((base/'terminal.json').read_text())
print('TERMINAL', json.dumps(t), flush=True)
print('FAILURE', (base/'failure.json').read_text() if (base/'failure.json').exists() else 'none', flush=True)
assert t['native_verified'] is True, 'native not verified'
m = base/'export'/'manifest.json'
print('A01_EXPORT', {k: v for k, v in json.loads(m.read_text()).items() if k != 'members'} if m.exists() else 'absent', flush=True)
PY
out=/mnt/local/_outputs/deep-llms_th2
receipt=$(sha256sum $out/dot_sandwich_20261123_a01/native-verified.json | cut -d' ' -f1)
echo "NATIVE_RECEIPT $receipt"
# 2) Launch DOT sandwich a02 (zero-init recipient ids).
pkg="$PWD/resources/dot_sandwich_a02_20261123.tar.xz"
echo "3bd5bde6dbefcff8d84939cac3c5e62ec3fca60f8d08c3437bc0c01cfe45074b  $pkg" | sha256sum -c -
extra=$(mktemp -d $out/dot_sandwich_a02_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
export PYTHONPATH=$out/dot_sandwich_20261123_a01/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/dot_sandwich_20261123_a02 \
  --old $out/dot_sandwich_20261123_a01/execution --prior-sha256 7b4a0f9c945e806f938b2a2a718ce6a261c9fb432cc8d8be92ddfba4c817ed64 \
  --added efficient_llm/warm_order/dot_sandwich.py scripts/loop_screen_study.py scripts/dot_sandwich_study.py \
  --replaced efficient_llm/warm_order/dot_sandwich.py scripts/loop_screen_study.py scripts/dot_sandwich_study.py \
  --module scripts.dot_sandwich_study \
  --handoff $out/dot_sandwich_20261123_a01/native-verified.json \
  --handoff-sha256 "$receipt"
/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_sandwich_20261123_a02')
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
