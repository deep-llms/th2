#1 +240+a
#th2-ohmv-dot-vs-dense8-2x-f12-20261126
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
date -u
out=/mnt/local/_outputs/deep-llms_th2
if [ -e "$out/dot_vs_dense8_2x_20261126_f12" ]; then echo F12_ALREADY_LAUNCHED; cat $out/dot_vs_dense8_2x_20261126_f12/status.json || true; exit 0; fi
/usr/bin/python3 - <<'PY'
import json
from pathlib import Path
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_vs_dense8_20261126_f11')
t = json.loads((base/'terminal.json').read_text())
print('F11_TERMINAL', json.dumps(t), flush=True)
assert t['native_verified'] is True, 'native not verified'
PY
receipt=$(sha256sum $out/dot_vs_dense8_20261126_f11/native-verified.json | cut -d' ' -f1)
echo "NATIVE_RECEIPT $receipt"
[ "$receipt" = 4e33d05d46c50187b160f8b6f6b326a6282b58deeefddd2ce91a7c8fc7becfa0 ]
prior=b2b6983651e7f47ffc8fbc71f1ddb591e3762c0f05ad4037559f1b21792bd8cd
pkg="$PWD/resources/dot_f12_20261126.tar.xz"
echo "58b12f63bd0776e059b404d621a0c509a2b37fe5bc45b88d4380198c5773ae4c  $pkg" | sha256sum -c -
extra=$(mktemp -d $out/dot_f12_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
export PYTHONPATH=$out/dot_vs_dense8_20261126_f11/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/dot_vs_dense8_2x_20261126_f12 \
  --old $out/dot_vs_dense8_20261126_f11/execution --prior-sha256 "$prior" \
  --added efficient_llm/warm_order/dot_stop_sandwich.py efficient_llm/warm_order/loop_screen.py scripts/dot_sandwich_study.py \
  --replaced efficient_llm/warm_order/dot_stop_sandwich.py efficient_llm/warm_order/loop_screen.py scripts/dot_sandwich_study.py \
  --module scripts.dot_sandwich_study \
  --handoff $out/dot_vs_dense8_20261126_f11/native-verified.json \
  --handoff-sha256 "$receipt"
/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_vs_dense8_2x_20261126_f12')
jobs = [(0, 'swst_d7_tmlp_ff512_lr14_fa_2x'), (1, 'swst_d7_tmlp_ff512_lr14_fa_2x'), (0, 'dense8_fa_2x'), (1, 'dense8_fa_2x'),
        (0, 'swst_d8_tmlp_ff512_lr14_fa_2x'), (1, 'swst_d8_tmlp_ff512_lr14_fa_2x'), (0, 'dense8_lr07_fa_2x'), (1, 'dense8_lr07_fa_2x')]
paths = {f's{s}/{a}': base/f's{s}'/a/'progress.json' for s, a in jobs}
for _ in range(36):
    if (base/'failure.json').is_file():
        break
    if len([p for p in paths.values() if p.is_file()]) == len(paths):
        break
    time.sleep(5)
print('F12_FAILURE', (base/'failure.json').read_text()[-3000:] if (base/'failure.json').is_file() else 'none', flush=True)
print('F12_PROGRESS', json.dumps({a: json.loads(p.read_text()) for a, p in paths.items() if p.is_file()}), flush=True)
print('F12_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
