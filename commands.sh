#1 +240+a
#th2-ohmv-wait-a03-then-dot-sandwich-a04-20261124
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
date -u
out=/mnt/local/_outputs/deep-llms_th2
if [ -e "$out/dot_sandwich_20261124_a04" ]; then echo A04_ALREADY_LAUNCHED; cat $out/dot_sandwich_20261124_a04/status.json || true; exit 0; fi
# 1) No signal: wait for DOT sandwich a03 to end on its own (downstream, native restore + verification).
/usr/bin/python3 - <<'PY'
import json, time
from pathlib import Path
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_sandwich_20261123_a03')
for i in range(720):  # up to 60 min
    if (base/'terminal.json').exists():
        break
    if i % 120 == 0:
        st = json.loads((base/'status.json').read_text()) if (base/'status.json').exists() else None
        print('WAITING', time.strftime('%H:%M:%S'), json.dumps(st), flush=True)
    time.sleep(5)
t = json.loads((base/'terminal.json').read_text())
print('A03_TERMINAL', json.dumps(t), flush=True)
print('A03_FAILURE', (base/'failure.json').read_text() if (base/'failure.json').exists() else 'none', flush=True)
for n in ('train_results_manifest.json', 'downstream_results_manifest.json'):
    m = base/'export'/n
    print(n, {k: v for k, v in json.loads(m.read_text()).items() if k != 'members'} if m.exists() else 'absent', flush=True)
assert t['native_verified'] is True, 'native not verified'
PY
receipt=$(sha256sum $out/dot_sandwich_20261123_a03/native-verified.json | cut -d' ' -f1)
echo "NATIVE_RECEIPT $receipt"
# 2) Launch DOT sandwich a04 (old source = a02 execution, whose source.json hash is pinned).
pkg="$PWD/resources/dot_sandwich_a04_20261124.tar.xz"
echo "8a606e6af587bc8b0cba5cac61da4e4c7ce010d0506e4939eec04d191b59103b  $pkg" | sha256sum -c -
extra=$(mktemp -d $out/dot_sandwich_a04_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
export PYTHONPATH=$out/dot_sandwich_20261123_a02/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/dot_sandwich_20261124_a04 \
  --old $out/dot_sandwich_20261123_a02/execution --prior-sha256 5ec4dc7bef68f0829e35e193255127fbbda8ba7906b6527c245242e15a2d4d69 \
  --added efficient_llm/warm_order/dot_sandwich.py efficient_llm/warm_order/loop_screen.py efficient_llm/warm_order/loop_screen_downstream.py scripts/loop_screen_study.py scripts/dot_sandwich_study.py \
  --replaced efficient_llm/warm_order/dot_sandwich.py efficient_llm/warm_order/loop_screen.py efficient_llm/warm_order/loop_screen_downstream.py scripts/loop_screen_study.py scripts/dot_sandwich_study.py \
  --module scripts.dot_sandwich_study \
  --handoff $out/dot_sandwich_20261123_a03/native-verified.json \
  --handoff-sha256 "$receipt"
/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_sandwich_20261124_a04')
jobs = [(0, 'sw_prior'), (1, 'sw_prior'), (0, 'sw6_route'), (1, 'sw6_route'), (0, 'sw6_prior'), (1, 'sw6_prior'), (0, 'sw_cf_soft'), (0, 'sw_rl2')]
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
