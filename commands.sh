#1 +240+a
#th2-ohmv-dot-lr-high-f08-20261125
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
date -u
out=/mnt/local/_outputs/deep-llms_th2
if [ -e "$out/dot_lr_high_20261125_f08" ]; then echo F08_ALREADY_LAUNCHED; cat $out/dot_lr_high_20261125_f08/status.json || true; exit 0; fi
/usr/bin/python3 - <<'PY'
import json
from pathlib import Path
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_lr_fairness_20261125_f07')
t = json.loads((base/'terminal.json').read_text())
print('F07_TERMINAL', json.dumps(t), flush=True)
assert t['native_verified'] is True, 'native not verified'
PY
receipt=$(sha256sum $out/dot_lr_fairness_20261125_f07/native-verified.json | cut -d' ' -f1)
echo "NATIVE_RECEIPT $receipt"
[ "$receipt" = d20034992ca8b8ca19d91e17409be0b9c87a66ec61b1894ebcf12d43fbadd50b ]
prior=6f7186e24f37516fb3848af0a3355a70818e2e62d836f84d023aa02232dd40b5
pkg="$PWD/resources/dot_f08_20261125.tar.xz"
echo "fe26ad0dcd51c3e90902a70d5397d058dcce27d0b8eb54f2c48c8757c9e6fa0c  $pkg" | sha256sum -c -
extra=$(mktemp -d $out/dot_f08_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
export PYTHONPATH=$out/dot_lr_fairness_20261125_f07/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/dot_lr_high_20261125_f08 \
  --old $out/dot_lr_fairness_20261125_f07/execution --prior-sha256 "$prior" \
  --added efficient_llm/warm_order/loop_screen.py scripts/dot_sandwich_study.py \
  --replaced efficient_llm/warm_order/loop_screen.py scripts/dot_sandwich_study.py \
  --module scripts.dot_sandwich_study \
  --handoff $out/dot_lr_fairness_20261125_f07/native-verified.json \
  --handoff-sha256 "$receipt"
/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_lr_high_20261125_f08')
jobs = [(0, 'loop_cycle_a11_lr14'), (1, 'loop_cycle_a11_lr14'), (0, 'swst_d6_tmlp_lr14'), (1, 'swst_d6_tmlp_lr14'),
        (0, 'loop_cycle_a11_lr20'), (0, 'swst_d6_tmlp_lr20'), (1, 'loop_cycle_a11_lr07'), (1, 'swst_d6_tmlp_lr07')]
paths = {f's{s}/{a}': base/f's{s}'/a/'progress.json' for s, a in jobs}
for _ in range(36):
    if (base/'failure.json').is_file():
        break
    if len([p for p in paths.values() if p.is_file()]) == len(paths):
        break
    time.sleep(5)
print('F08_FAILURE', (base/'failure.json').read_text()[-3000:] if (base/'failure.json').is_file() else 'none', flush=True)
print('F08_PROGRESS', json.dumps({a: json.loads(p.read_text()) for a, p in paths.items() if p.is_file()}), flush=True)
print('F08_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
