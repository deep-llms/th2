#1 +240+a
#th2-ohmv-dot-depth-scale-f05-20261125
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
date -u
out=/mnt/local/_outputs/deep-llms_th2
if [ -e "$out/dot_depth_scale_20261125_f05" ]; then echo F05_ALREADY_LAUNCHED; cat $out/dot_depth_scale_20261125_f05/status.json || true; exit 0; fi
/usr/bin/python3 - <<'PY'
import json
from pathlib import Path
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_tmlp_20261125_f04')
t = json.loads((base/'terminal.json').read_text())
print('F04_TERMINAL', json.dumps(t), flush=True)
assert t['native_verified'] is True, 'native not verified'
PY
receipt=$(sha256sum $out/dot_tmlp_20261125_f04/native-verified.json | cut -d' ' -f1)
echo "NATIVE_RECEIPT $receipt"
[ "$receipt" = b05fd67257de15e64c197183654f38642a6941255ba0296af59ff132274ffc61 ]
prior=52e2ab0550448ac5915158a154c3090654793578cebb26cc656c9edbda08468b
pkg="$PWD/resources/dot_f05_20261125.tar.xz"
echo "1d10c190e8196931c5792f69bb82f8d6868344b8701c6d523016d7a9f119ccb4  $pkg" | sha256sum -c -
extra=$(mktemp -d $out/dot_f05_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
export PYTHONPATH=$out/dot_tmlp_20261125_f04/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/dot_depth_scale_20261125_f05 \
  --old $out/dot_tmlp_20261125_f04/execution --prior-sha256 "$prior" \
  --added efficient_llm/warm_order/loop_baselines.py efficient_llm/warm_order/loop_screen.py scripts/dot_sandwich_study.py \
  --replaced efficient_llm/warm_order/loop_baselines.py efficient_llm/warm_order/loop_screen.py scripts/dot_sandwich_study.py \
  --module scripts.dot_sandwich_study \
  --handoff $out/dot_tmlp_20261125_f04/native-verified.json \
  --handoff-sha256 "$receipt"
/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_depth_scale_20261125_f05')
jobs = [(0, 'swst_d5_tmlp'), (1, 'swst_d5_tmlp'), (0, 'loop_cycle_a9'), (1, 'loop_cycle_a9'),
        (0, 'swst_d4_h4'), (1, 'swst_d4_h4'), (0, 'loop_prc_a8'), (1, 'loop_prc_a8')]
paths = {f's{s}/{a}': base/f's{s}'/a/'progress.json' for s, a in jobs}
for _ in range(36):
    if (base/'failure.json').is_file():
        break
    if len([p for p in paths.values() if p.is_file()]) == len(paths):
        break
    time.sleep(5)
print('F05_FAILURE', (base/'failure.json').read_text()[-3000:] if (base/'failure.json').is_file() else 'none', flush=True)
print('F05_PROGRESS', json.dumps({a: json.loads(p.read_text()) for a, p in paths.items() if p.is_file()}), flush=True)
print('F05_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
