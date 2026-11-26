#1 +240+a
#th2-ohmv-dot-ffn-control-f16-20261126
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
date -u
out=/mnt/local/_outputs/deep-llms_th2
if [ -e "$out/dot_ffn_control_20261126_f16" ]; then echo F16_ALREADY_LAUNCHED; cat $out/dot_ffn_control_20261126_f16/status.json || true; exit 0; fi
/usr/bin/python3 - <<'PY'
import json
from pathlib import Path
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_f15_20261126')
t = json.loads((base/'terminal.json').read_text())
print('F15_TERMINAL', json.dumps(t), flush=True)
assert t['native_verified'] is True, 'native not verified'
PY
receipt=$(sha256sum $out/dot_f15_20261126/native-verified.json | cut -d' ' -f1)
echo "NATIVE_RECEIPT $receipt"
[ "$receipt" = d84ef2853034f36ea11fa4dd9ad573f20fb863bed993e7964225b15345876d81 ]
prior=0cb74a361389444606137347d821400b4efeaecfa66ce9f1f4e7aee4f5c99e50
pkg="$PWD/resources/dot_f16_20261126.tar.xz"
echo "445e6030909a394e71051172134f8d08e95de5e3ee8cb551b7eb2a447a1f627f  $pkg" | sha256sum -c -
extra=$(mktemp -d $out/dot_f16_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
export PYTHONPATH=$out/dot_f15_20261126/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/dot_ffn_control_20261126_f16 \
  --old $out/dot_f15_20261126/execution --prior-sha256 "$prior" \
  --added efficient_llm/warm_order/loop_baselines.py efficient_llm/warm_order/loop_screen.py scripts/dot_sandwich_study.py \
  --replaced efficient_llm/warm_order/loop_baselines.py efficient_llm/warm_order/loop_screen.py scripts/dot_sandwich_study.py \
  --module scripts.dot_sandwich_study \
  --handoff $out/dot_f15_20261126/native-verified.json \
  --handoff-sha256 "$receipt"
/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_ffn_control_20261126_f16')
jobs = [(0, 'dense8_ff2624_fa'), (1, 'dense8_ff2624_fa'), (0, 'loop_cycle_a8_ff2176_fa'), (1, 'loop_cycle_a8_ff2176_fa'),
        (0, 'loop_cycle_a7_ff2624_fa'), (1, 'loop_cycle_a7_ff2624_fa'), (0, 'swst_m8d8_tmlp_ff512_fa_2x'), (1, 'swst_m8d8_tmlp_ff512_fa_2x')]
paths = {f's{s}/{a}': base/f's{s}'/a/'progress.json' for s, a in jobs}
for _ in range(36):
    if (base/'failure.json').is_file():
        break
    if len([p for p in paths.values() if p.is_file()]) == len(paths):
        break
    time.sleep(5)
print('F16_FAILURE', (base/'failure.json').read_text()[-3000:] if (base/'failure.json').is_file() else 'none', flush=True)
print('F16_PROGRESS', json.dumps({a: json.loads(p.read_text()) for a, p in paths.items() if p.is_file()}), flush=True)
print('F16_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
