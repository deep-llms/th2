#1 +240+a
#th2-ohmv-dot-sandwich-recipe-lm-a07-20261125
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
date -u
out=/mnt/local/_outputs/deep-llms_th2
if [ -e "$out/dot_recipe_lm_20261125_a07" ]; then echo A07_ALREADY_LAUNCHED; cat $out/dot_recipe_lm_20261125_a07/status.json || true; exit 0; fi
/usr/bin/python3 - <<'PY'
import json
from pathlib import Path
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_synth_20261125_s11')
t = json.loads((base/'terminal.json').read_text())
print('S11_TERMINAL', json.dumps(t), flush=True)
assert t['native_verified'] is True, 'native not verified'
PY
receipt=$(sha256sum $out/dot_synth_20261125_s11/native-verified.json | cut -d' ' -f1)
echo "NATIVE_RECEIPT $receipt"
pkg="$PWD/resources/dot_lm_a07_20261125.tar.xz"
echo "2913b01b93bfe074cccf2f2dfb6d939573c16464b2e0a4217f4c53620568827c  $pkg" | sha256sum -c -
extra=$(mktemp -d $out/dot_lm_a07_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
export PYTHONPATH=$out/dot_synth_20261125_s11/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/dot_recipe_lm_20261125_a07 \
  --old $out/dot_synth_20261125_s11/execution --prior-sha256 f7be34ec52c6c1ce1e2ff5b7e46f4554fc7c76b10548e36e47a1a2562abc253d \
  --added efficient_llm/warm_order/dot_stop_sandwich.py efficient_llm/warm_order/loop_screen.py scripts/loop_screen_study.py scripts/dot_sandwich_study.py \
  --replaced efficient_llm/warm_order/dot_stop_sandwich.py efficient_llm/warm_order/loop_screen.py scripts/loop_screen_study.py scripts/dot_sandwich_study.py \
  --module scripts.dot_sandwich_study \
  --handoff $out/dot_synth_20261125_s11/native-verified.json \
  --handoff-sha256 "$receipt"
/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_recipe_lm_20261125_a07')
jobs = [(0, 'swst_fixed_d4r'), (1, 'swst_fixed_d4r'), (0, 'swst_loop1_d4r'), (1, 'swst_loop1_d4r'),
        (0, 'swst_prior_d4gr'), (1, 'swst_prior_d4gr'), (1, 'swst_fixed_d4'), (0, 'swst_fixed_d4r8')]
paths = {f's{s}/{a}': base/f's{s}'/a/'progress.json' for s, a in jobs}
for _ in range(36):
    if (base/'failure.json').is_file():
        break
    if len([p for p in paths.values() if p.is_file()]) == len(paths):
        break
    time.sleep(5)
print('A07_FAILURE', (base/'failure.json').read_text()[-3000:] if (base/'failure.json').is_file() else 'none', flush=True)
print('A07_PROGRESS', json.dumps({a: json.loads(p.read_text()) for a, p in paths.items() if p.is_file()}), flush=True)
print('A07_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
