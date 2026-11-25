#1 +240+a
#th2-ohmv-dot-fixed-improve-f01-20261125
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
date -u
out=/mnt/local/_outputs/deep-llms_th2
if [ -e "$out/dot_fixed_improve_20261125_f01" ]; then echo F01_ALREADY_LAUNCHED; cat $out/dot_fixed_improve_20261125_f01/status.json || true; exit 0; fi
/usr/bin/python3 - <<'PY'
import json
from pathlib import Path
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_t_loops_20261125_t01')
t = json.loads((base/'terminal.json').read_text())
print('T01_TERMINAL', json.dumps(t), flush=True)
assert t['native_verified'] is True, 'native not verified'
PY
receipt=$(sha256sum $out/dot_t_loops_20261125_t01/native-verified.json | cut -d' ' -f1)
echo "NATIVE_RECEIPT $receipt"
pkg="$PWD/resources/dot_f01_20261125.tar.xz"
echo "98774ee44053ae041d5b065edd53534e7b760644fc43cce4ca8b03bcce03adf9  $pkg" | sha256sum -c -
extra=$(mktemp -d $out/dot_f01_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
export PYTHONPATH=$out/dot_t_loops_20261125_t01/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/dot_fixed_improve_20261125_f01 \
  --old $out/dot_t_loops_20261125_t01/execution --prior-sha256 3c5bc6b4264c68531720d5ec04fa964ae4ea47f60ac974c070f10c43fa4d461b \
  --added efficient_llm/warm_order/dot_stop_sandwich.py efficient_llm/warm_order/loop_screen.py scripts/loop_screen_study.py scripts/dot_sandwich_study.py \
  --replaced efficient_llm/warm_order/dot_stop_sandwich.py efficient_llm/warm_order/loop_screen.py scripts/loop_screen_study.py scripts/dot_sandwich_study.py \
  --module scripts.dot_sandwich_study \
  --handoff $out/dot_t_loops_20261125_t01/native-verified.json \
  --handoff-sha256 "$receipt"
/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_fixed_improve_20261125_f01')
jobs = [(0, 'swst_fixed_d3_film'), (1, 'swst_fixed_d3_film'), (0, 'swst_fixed_d3_lora'), (1, 'swst_fixed_d3_lora'),
        (0, 'swst_fixed_d4_light'), (1, 'swst_fixed_d4_light'), (0, 'swst_fixed_d3_fl'), (0, 'swst_fixed_d4_lightf')]
paths = {f's{s}/{a}': base/f's{s}'/a/'progress.json' for s, a in jobs}
for _ in range(36):
    if (base/'failure.json').is_file():
        break
    if len([p for p in paths.values() if p.is_file()]) == len(paths):
        break
    time.sleep(5)
print('F01_FAILURE', (base/'failure.json').read_text()[-3000:] if (base/'failure.json').is_file() else 'none', flush=True)
print('F01_PROGRESS', json.dumps({a: json.loads(p.read_text()) for a, p in paths.items() if p.is_file()}), flush=True)
print('F01_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
