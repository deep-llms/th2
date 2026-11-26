#1 +240+a
#th2-ohmv-dot-f20-20261126
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
date -u
out=/mnt/local/_outputs/deep-llms_th2
if [ -e "$out/dot_f20_20261126" ]; then echo F20_ALREADY_LAUNCHED; cat $out/dot_f20_20261126/status.json || true; exit 0; fi
/usr/bin/python3 - <<'PY'
import json
from pathlib import Path
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_f19_20261126')
t = json.loads((base/'terminal.json').read_text())
print('F19_TERMINAL', json.dumps(t), flush=True)
assert t['native_verified'] is True and t['success'] is True, 'f19 did not finish with the native burn verified'
PY
receipt=4ef96c478448ee0af163d5a658ba13c5f0f132ca41e3edaef850d5185cadf7f0
echo "$receipt  $out/dot_f19_20261126/native-verified.json" | sha256sum -c -
prior=397438ad5e7320c2169d1d19ef2c15acbb7a555f0dd21a4497354003b3c73465
pkg="$PWD/resources/dot_f20_20261126.tar.xz"
echo "424c933d9ac8c99d976cb8c309f4350ff689809b78861f8d60a742836f93b0ef  $pkg" | sha256sum -c -
extra=$(mktemp -d $out/dot_f20_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
export PYTHONPATH=$out/dot_f19_20261126/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/dot_f20_20261126 \
  --old $out/dot_f19_20261126/execution --prior-sha256 "$prior" \
  --added efficient_llm/warm_order/loop_screen.py scripts/dot_sandwich_study.py \
  --replaced efficient_llm/warm_order/loop_screen.py scripts/dot_sandwich_study.py \
  --module scripts.dot_sandwich_study \
  --handoff $out/dot_f19_20261126/native-verified.json \
  --handoff-sha256 "$receipt"
/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_f20_20261126')
jobs = [(0, 'huginn_r3_lr14'), (0, 'huginn_r3_lr20'), (0, 'huginn_r2'), (0, 'ouro_t2_lr14'), (0, 'ouro_t2_lr20'),
        (0, 'mor_mc3_lr07'), (0, 'mor_mc3_lr05'), (0, 'mor_mc2')]
paths = {f's{s}/{a}': base/f's{s}'/a/'progress.json' for s, a in jobs}
for _ in range(48):
    if (base/'failure.json').is_file():
        break
    if len([p for p in paths.values() if p.is_file()]) == len(paths):
        break
    time.sleep(5)
print('F20_FAILURE', (base/'failure.json').read_text()[-3000:] if (base/'failure.json').is_file() else 'none', flush=True)
print('F20_PROGRESS', json.dumps({a: json.loads(p.read_text()) for a, p in paths.items() if p.is_file()}), flush=True)
print('F20_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
