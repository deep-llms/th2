#1 +240+a
#th2-ohmv-wait-headroom2-then-dot-stop-a06-20261124
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
date -u
out=/mnt/local/_outputs/deep-llms_th2
if [ -e "$out/dot_stop_20261124_a06" ]; then echo A06_ALREADY_LAUNCHED; cat $out/dot_stop_20261124_a06/status.json || true; exit 0; fi
# 1) No signal: wait for the learnable-headroom diagnostic to end on its own; print its results.
/usr/bin/python3 - <<'PY'
import json, time
from pathlib import Path
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_headroom2_20261124_a01')
for i in range(360):  # up to 30 min
    if (base/'terminal.json').exists():
        break
    time.sleep(5)
t = json.loads((base/'terminal.json').read_text())
print('H2_TERMINAL', json.dumps(t), flush=True)
print('H2_FAILURE', (base/'failure.json').read_text()[-1500:] if (base/'failure.json').exists() else 'none', flush=True)
for p in sorted(base.glob('*/summary.json')):
    print('HEADROOM2', p.read_text(), flush=True)
m = base/'export'/'manifest.json'
print('H2_EXPORT', {k: v for k, v in json.loads(m.read_text()).items() if k != 'members'} if m.exists() else 'absent', flush=True)
assert t['native_verified'] is True, 'native not verified'
PY
receipt=$(sha256sum $out/dot_headroom2_20261124_a01/native-verified.json | cut -d' ' -f1)
echo "NATIVE_RECEIPT $receipt"
# 2) Launch DOT stop sandwich a06 (old source = the first headroom execution, pinned hash).
pkg="$PWD/resources/dot_stop_a06_20261124.tar.xz"
echo "41cc778ad789517f25456bff08414dfc4f5e1b321eb445713c52120b07c3c533  $pkg" | sha256sum -c -
extra=$(mktemp -d $out/dot_stop_a06_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
export PYTHONPATH=$out/dot_headroom_20261124_a01/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/dot_stop_20261124_a06 \
  --old $out/dot_headroom_20261124_a01/execution --prior-sha256 cfaf46bb7628876292550f3d4efaa3767d0affaca66712faed776be587924bae \
  --added efficient_llm/warm_order/dot_stop_sandwich.py efficient_llm/warm_order/dot_sandwich.py efficient_llm/warm_order/loop_screen.py scripts/loop_screen_study.py scripts/dot_sandwich_study.py \
  --replaced efficient_llm/warm_order/dot_sandwich.py efficient_llm/warm_order/loop_screen.py scripts/loop_screen_study.py scripts/dot_sandwich_study.py \
  --module scripts.dot_sandwich_study \
  --handoff $out/dot_headroom2_20261124_a01/native-verified.json \
  --handoff-sha256 "$receipt"
/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_stop_20261124_a06')
jobs = [(0, 'swst_dot_l02'), (1, 'swst_dot_l02'), (0, 'swst_dot_l05'), (0, 'swst_fixed_l02'), (1, 'swst_fixed_l02'),
        (0, 'swst_budget_l02'), (0, 'swst_route_d4'), (0, 'swst_fixed_d4')]
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
