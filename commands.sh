#1 +240+a
#th2-ohmv-dot-vs-dense8-f11-20261126
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
date -u
out=/mnt/local/_outputs/deep-llms_th2
if [ -e "$out/dot_vs_dense8_20261126_f11" ]; then echo F11_ALREADY_LAUNCHED; cat $out/dot_vs_dense8_20261126_f11/status.json || true; exit 0; fi
/usr/bin/python3 - <<'PY'
import json
from pathlib import Path
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_fast_attn_20261125_f10')
t = json.loads((base/'terminal.json').read_text())
print('F10_TERMINAL', json.dumps(t), flush=True)
assert t['native_verified'] is True, 'native not verified'
PY
receipt=$(sha256sum $out/dot_fast_attn_20261125_f10/native-verified.json | cut -d' ' -f1)
echo "NATIVE_RECEIPT $receipt"
[ "$receipt" = d25a7049f81ddfd50cc826588d378e152ca565f8980fb239f4d1e76f590d4fcf ]
prior=cdcafe03f6fd6a6595742d1180fd8e0a576d6783c4f03e7d0abfb26a8de2cf4d
pkg="$PWD/resources/dot_f11_20261126.tar.xz"
echo "e5dccf75c849dd7ea6d9a0b70116d1f7b590db9b26b1233db1be420af428da17  $pkg" | sha256sum -c -
extra=$(mktemp -d $out/dot_f11_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
export PYTHONPATH=$out/dot_fast_attn_20261125_f10/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/dot_vs_dense8_20261126_f11 \
  --old $out/dot_fast_attn_20261125_f10/execution --prior-sha256 "$prior" \
  --added efficient_llm/warm_order/loop_screen.py scripts/dot_sandwich_study.py \
  --replaced efficient_llm/warm_order/loop_screen.py scripts/dot_sandwich_study.py \
  --module scripts.dot_sandwich_study \
  --handoff $out/dot_fast_attn_20261125_f10/native-verified.json \
  --handoff-sha256 "$receipt"
/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_vs_dense8_20261126_f11')
jobs = [(0, 'swst_d7_tmlp_ff512_lr14_fa'), (1, 'swst_d7_tmlp_ff512_lr14_fa'), (0, 'swst_d7_tmlp_ff512_fa'), (1, 'swst_d7_tmlp_ff512_fa'),
        (0, 'swst_d8_tmlp_ff512_lr14_fa'), (1, 'swst_d8_tmlp_ff512_lr14_fa'), (0, 'dense8_lr07_fa'), (1, 'dense8_lr07_fa')]
paths = {f's{s}/{a}': base/f's{s}'/a/'progress.json' for s, a in jobs}
for _ in range(36):
    if (base/'failure.json').is_file():
        break
    if len([p for p in paths.values() if p.is_file()]) == len(paths):
        break
    time.sleep(5)
print('F11_FAILURE', (base/'failure.json').read_text()[-3000:] if (base/'failure.json').is_file() else 'none', flush=True)
print('F11_PROGRESS', json.dumps({a: json.loads(p.read_text()) for a, p in paths.items() if p.is_file()}), flush=True)
print('F11_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
