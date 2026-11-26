#1 +240+a
#th2-ohmv-dot-f22-20261126
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
date -u
out=/mnt/local/_outputs/deep-llms_th2
if [ -e "$out/dot_f22_20261126" ]; then echo F22_ALREADY_LAUNCHED; cat $out/dot_f22_20261126/status.json || true; exit 0; fi
/usr/bin/python3 - <<'PY'
import json
from pathlib import Path
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_f21_20261126')
t = json.loads((base/'terminal.json').read_text())
print('F21_TERMINAL', json.dumps(t), flush=True)
assert t['native_verified'] is True and t['success'] is True, 'f21 did not finish with the native burn verified'
PY
receipt=4574e351b91279147835811c2cafbff4f3826686cb7bc85983c215888d0b5e92
echo "$receipt  $out/dot_f21_20261126/native-verified.json" | sha256sum -c -
prior=d3f135d26f65ea75a88e29fb6369e5e6e40f818f0913a28ce764064ffacd0437
pkg="$PWD/resources/dot_f22_20261126.tar.xz"
echo "13c4ddbaaec0086080f18e57b0e0d497abb6b6cf1d30469f874e4112b3a0559e  $pkg" | sha256sum -c -
extra=$(mktemp -d $out/dot_f22_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
export PYTHONPATH=$out/dot_f21_20261126/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/dot_f22_20261126 \
  --old $out/dot_f21_20261126/execution --prior-sha256 "$prior" \
  --added efficient_llm/warm_order/full_methods.py efficient_llm/warm_order/loop_screen.py scripts/dot_sandwich_study.py \
  --replaced efficient_llm/warm_order/full_methods.py efficient_llm/warm_order/loop_screen.py scripts/dot_sandwich_study.py \
  --module scripts.dot_sandwich_study \
  --handoff $out/dot_f21_20261126/native-verified.json \
  --handoff-sha256 "$receipt"
/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_f22_20261126')
jobs = [(1, 'huginn_r3_lr28'), (0, 'huginn_r2_lr28'), (0, 'huginn_r3_lr20_prenorm'), (0, 'huginn_r3_lr20_adamw'),
        (0, 'huginn_r3_lr20_zero_state'), (0, 'huginn_r3_lr20_fixed_r'), (0, 'huginn_r3_lr20_add_inject'),
        (0, 'huginn_r3_lr20_no_emb_scale')]
paths = {f's{s}/{a}': base/f's{s}'/a/'progress.json' for s, a in jobs}
for _ in range(48):
    if (base/'failure.json').is_file():
        break
    if len([p for p in paths.values() if p.is_file()]) == len(paths):
        break
    time.sleep(5)
print('F22_FAILURE', (base/'failure.json').read_text()[-3000:] if (base/'failure.json').is_file() else 'none', flush=True)
print('F22_PROGRESS', json.dumps({a: json.loads(p.read_text()) for a, p in paths.items() if p.is_file()}), flush=True)
print('F22_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
