#1 +240+a
#th2-ohmv-dot-f21-20261126
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
date -u
out=/mnt/local/_outputs/deep-llms_th2
if [ -e "$out/dot_f21_20261126" ]; then echo F21_ALREADY_LAUNCHED; cat $out/dot_f21_20261126/status.json || true; exit 0; fi
/usr/bin/python3 - <<'PY'
import json
from pathlib import Path
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_f20_20261126')
t = json.loads((base/'terminal.json').read_text())
print('F20_TERMINAL', json.dumps(t), flush=True)
assert t['native_verified'] is True and t['success'] is True, 'f20 did not finish with the native burn verified'
PY
receipt=082615afde0494cc0f138d5a1179aff3344b89809bcc80a699f0ed6da88dc137
echo "$receipt  $out/dot_f20_20261126/native-verified.json" | sha256sum -c -
prior=f0dcc7946b5ba2ed67ba5511b010a6e8f7f551cabafb568a0da1c8b98e2fad53
pkg="$PWD/resources/dot_f21_20261126.tar.xz"
echo "3a104b5de06617e22be53b7a2a0d90e550b23e434e2dc8b94a74fe498de362b7  $pkg" | sha256sum -c -
extra=$(mktemp -d $out/dot_f21_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
export PYTHONPATH=$out/dot_f20_20261126/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/dot_f21_20261126 \
  --old $out/dot_f20_20261126/execution --prior-sha256 "$prior" \
  --added efficient_llm/warm_order/loop_screen.py scripts/dot_sandwich_study.py \
  --replaced efficient_llm/warm_order/loop_screen.py scripts/dot_sandwich_study.py \
  --module scripts.dot_sandwich_study \
  --handoff $out/dot_f20_20261126/native-verified.json \
  --handoff-sha256 "$receipt"
/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_f21_20261126')
jobs = [(0, 'huginn_r3_lr28'), (0, 'huginn_r3_lr40'), (0, 'huginn_r2_lr20'), (1, 'huginn_r2_lr20'), (1, 'huginn_r3_lr20'),
        (1, 'ouro_t2_lr14'), (1, 'mor_mc3'), (0, 'swst_d7_tmlp_ff512_unshared_lr20_fa')]
paths = {f's{s}/{a}': base/f's{s}'/a/'progress.json' for s, a in jobs}
for _ in range(48):
    if (base/'failure.json').is_file():
        break
    if len([p for p in paths.values() if p.is_file()]) == len(paths):
        break
    time.sleep(5)
print('F21_FAILURE', (base/'failure.json').read_text()[-3000:] if (base/'failure.json').is_file() else 'none', flush=True)
print('F21_PROGRESS', json.dumps({a: json.loads(p.read_text()) for a, p in paths.items() if p.is_file()}), flush=True)
print('F21_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
