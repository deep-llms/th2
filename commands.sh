#1 +240+a
#th2-ohmv-dot-lr-fairness-f07-20261125
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
date -u
out=/mnt/local/_outputs/deep-llms_th2
if [ -e "$out/dot_lr_fairness_20261125_f07" ]; then echo F07_ALREADY_LAUNCHED; cat $out/dot_lr_fairness_20261125_f07/status.json || true; exit 0; fi
/usr/bin/python3 - <<'PY'
import json
from pathlib import Path
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_depth_scale_20261125_f06')
t = json.loads((base/'terminal.json').read_text())
print('F06_TERMINAL', json.dumps(t), flush=True)
assert t['native_verified'] is True, 'native not verified'
PY
receipt=$(sha256sum $out/dot_depth_scale_20261125_f06/native-verified.json | cut -d' ' -f1)
echo "NATIVE_RECEIPT $receipt"
[ "$receipt" = 9f1ee1e856c378761119d62be5e49708c4888f126c92145307af9c0943a95179 ]
prior=1a404fd9c3266a6ce3687c792c8624d8b12763cdbc196857cd573b1fa9a69b24
pkg="$PWD/resources/dot_f07_20261125.tar.xz"
echo "9d1c9569043599312ba2f16f0527d996f2f730fc92261f612db8886d336914b1  $pkg" | sha256sum -c -
extra=$(mktemp -d $out/dot_f07_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
export PYTHONPATH=$out/dot_depth_scale_20261125_f06/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/dot_lr_fairness_20261125_f07 \
  --old $out/dot_depth_scale_20261125_f06/execution --prior-sha256 "$prior" \
  --added efficient_llm/warm_order/dot_stop_sandwich.py efficient_llm/warm_order/loop_screen.py scripts/loop_screen_study.py scripts/dot_sandwich_study.py \
  --replaced efficient_llm/warm_order/dot_stop_sandwich.py efficient_llm/warm_order/loop_screen.py scripts/loop_screen_study.py scripts/dot_sandwich_study.py \
  --module scripts.dot_sandwich_study \
  --handoff $out/dot_depth_scale_20261125_f06/native-verified.json \
  --handoff-sha256 "$receipt"
/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_lr_fairness_20261125_f07')
jobs = [(0, 'loop_cycle_a11_lr05'), (1, 'loop_cycle_a11_lr05'), (0, 'swst_d6_tmlp_lr05'), (1, 'swst_d6_tmlp_lr05'),
        (0, 'loop_cycle_a11_lr07'), (0, 'swst_d6_tmlp_lr07'), (0, 'loop_cycle_a9_lr05'), (0, 'swst_d5_tmlp_lr05')]
paths = {f's{s}/{a}': base/f's{s}'/a/'progress.json' for s, a in jobs}
for _ in range(36):
    if (base/'failure.json').is_file():
        break
    if len([p for p in paths.values() if p.is_file()]) == len(paths):
        break
    time.sleep(5)
print('F07_FAILURE', (base/'failure.json').read_text()[-3000:] if (base/'failure.json').is_file() else 'none', flush=True)
print('F07_PROGRESS', json.dumps({a: json.loads(p.read_text()) for a, p in paths.items() if p.is_file()}), flush=True)
print('F07_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
