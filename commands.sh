#1 +240+a
#th2-ohmv-dot-vs-inkling-f17-20261126
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
date -u
out=/mnt/local/_outputs/deep-llms_th2
if [ -e "$out/dot_vs_inkling_20261126_f17" ]; then echo F17_ALREADY_LAUNCHED; cat $out/dot_vs_inkling_20261126_f17/status.json || true; exit 0; fi
/usr/bin/python3 - <<'PY'
import json
from pathlib import Path
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_ffn_control_20261126_f16')
t = json.loads((base/'terminal.json').read_text())
print('F16_TERMINAL', json.dumps(t), flush=True)
assert t['native_verified'] is True, 'native not verified'
PY
receipt=$(sha256sum $out/dot_ffn_control_20261126_f16/native-verified.json | cut -d' ' -f1)
echo "NATIVE_RECEIPT $receipt"
[ "$receipt" = 88836ba2e9e993efc63722fe16307865c578422ede2c706ced389f15e0450d99 ]
prior=c1b7e60f6101818f3774310aaec774e108751ed08e554811d7b03f0ae4fe0f97
pkg="$PWD/resources/dot_f17_20261126.tar.xz"
echo "a5f09a3f47adad8dfb5a1c8b9d8b8167a710888556952999d48410e9877402ab  $pkg" | sha256sum -c -
extra=$(mktemp -d $out/dot_f17_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
export PYTHONPATH=$out/dot_ffn_control_20261126_f16/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/dot_vs_inkling_20261126_f17 \
  --old $out/dot_ffn_control_20261126_f16/execution --prior-sha256 "$prior" \
  --added efficient_llm/warm_order/loop_screen.py scripts/dot_sandwich_study.py \
  --replaced efficient_llm/warm_order/loop_screen.py scripts/dot_sandwich_study.py \
  --module scripts.dot_sandwich_study \
  --handoff $out/dot_ffn_control_20261126_f16/native-verified.json \
  --handoff-sha256 "$receipt"
/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_vs_inkling_20261126_f17')
jobs = [(0, 'gated_fa'), (1, 'gated_fa'), (0, 'memory_fa'), (1, 'memory_fa'), (0, 'loop_attnres_fa'), (1, 'loop_attnres_fa'),
        (0, 'attnres8_fa'), (1, 'attnres8_fa')]
paths = {f's{s}/{a}': base/f's{s}'/a/'progress.json' for s, a in jobs}
for _ in range(36):
    if (base/'failure.json').is_file():
        break
    if len([p for p in paths.values() if p.is_file()]) == len(paths):
        break
    time.sleep(5)
print('F17_FAILURE', (base/'failure.json').read_text()[-3000:] if (base/'failure.json').is_file() else 'none', flush=True)
print('F17_PROGRESS', json.dumps({a: json.loads(p.read_text()) for a, p in paths.items() if p.is_file()}), flush=True)
print('F17_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
