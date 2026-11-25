#1 +240+a
#th2-ohmv-dot-depth-scale-f06-20261125
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
date -u
out=/mnt/local/_outputs/deep-llms_th2
if [ -e "$out/dot_depth_scale_20261125_f06" ]; then echo F06_ALREADY_LAUNCHED; cat $out/dot_depth_scale_20261125_f06/status.json || true; exit 0; fi
/usr/bin/python3 - <<'PY'
import json
from pathlib import Path
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_depth_scale_20261125_f05')
t = json.loads((base/'terminal.json').read_text())
print('F05_TERMINAL', json.dumps(t), flush=True)
assert t['native_verified'] is True, 'native not verified'
PY
receipt=$(sha256sum $out/dot_depth_scale_20261125_f05/native-verified.json | cut -d' ' -f1)
echo "NATIVE_RECEIPT $receipt"
[ "$receipt" = 52f02761ed7fc243d61326bb771df738b7c9d9d6f861c631c8acc7d897ad5b3a ]
prior=832e39ca6ebaa22b74a78b9e7011faa000590971b093188b855bf065b472d870
pkg="$PWD/resources/dot_f06_20261125.tar.xz"
echo "306e3ceda31b93774f0f223aabaa9d865a2d0d296e6289280516ea1290401fa2  $pkg" | sha256sum -c -
extra=$(mktemp -d $out/dot_f06_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
export PYTHONPATH=$out/dot_depth_scale_20261125_f05/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/dot_depth_scale_20261125_f06 \
  --old $out/dot_depth_scale_20261125_f05/execution --prior-sha256 "$prior" \
  --added efficient_llm/warm_order/loop_screen.py scripts/dot_sandwich_study.py \
  --replaced efficient_llm/warm_order/loop_screen.py scripts/dot_sandwich_study.py \
  --module scripts.dot_sandwich_study \
  --handoff $out/dot_depth_scale_20261125_f05/native-verified.json \
  --handoff-sha256 "$receipt"
/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_depth_scale_20261125_f06')
jobs = [(0, 'swst_d6_tmlp'), (1, 'swst_d6_tmlp'), (0, 'loop_cycle_a11'), (1, 'loop_cycle_a11'),
        (0, 'swst_d5_h4'), (1, 'swst_d5_h4'), (0, 'loop_cycle_a10'), (1, 'loop_cycle_a10')]
paths = {f's{s}/{a}': base/f's{s}'/a/'progress.json' for s, a in jobs}
for _ in range(36):
    if (base/'failure.json').is_file():
        break
    if len([p for p in paths.values() if p.is_file()]) == len(paths):
        break
    time.sleep(5)
print('F06_FAILURE', (base/'failure.json').read_text()[-3000:] if (base/'failure.json').is_file() else 'none', flush=True)
print('F06_PROGRESS', json.dumps({a: json.loads(p.read_text()) for a, p in paths.items() if p.is_file()}), flush=True)
print('F06_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
