#1 +240+a
#th2-ohmv-wait-f01-then-dot-compact-t-f02-20261125
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
date -u
out=/mnt/local/_outputs/deep-llms_th2
if [ -e "$out/dot_compact_t_20261125_f02" ]; then echo F02_ALREADY_LAUNCHED; cat $out/dot_compact_t_20261125_f02/status.json || true; exit 0; fi
/usr/bin/python3 - <<'PY'
import json
from pathlib import Path
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_fixed_improve_20261125_f01')
import time
for i in range(1320):  # no signal: wait up to 110 min for f01 (train + downstream) to end on its own
    if (base/'terminal.json').exists() or (base/'failure.json').exists():
        break
    time.sleep(5)
print('F01_FAILURE', (base/'failure.json').read_text()[-1500:] if (base/'failure.json').exists() else 'none', flush=True)
t = json.loads((base/'terminal.json').read_text())
print('F01_TERMINAL', json.dumps(t), flush=True)
assert t['native_verified'] is True, 'native not verified'
PY
receipt=$(sha256sum $out/dot_fixed_improve_20261125_f01/native-verified.json | cut -d' ' -f1)
echo "NATIVE_RECEIPT $receipt"
pkg="$PWD/resources/dot_f02_20261125.tar.xz"
echo "7f52bf8237006fae12162225287e0981f310105f06076628b9fc655bde0a52f8  $pkg" | sha256sum -c -
extra=$(mktemp -d $out/dot_f02_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
export PYTHONPATH=$out/dot_fixed_improve_20261125_f01/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/dot_compact_t_20261125_f02 \
  --old $out/dot_fixed_improve_20261125_f01/execution --prior-sha256 4a87411be92c287e95485bee8462808a51642c5774ec1c52224df446665802c9 \
  --added efficient_llm/warm_order/dot_stop_sandwich.py efficient_llm/warm_order/loop_screen.py scripts/loop_screen_study.py scripts/dot_sandwich_study.py \
  --replaced efficient_llm/warm_order/dot_stop_sandwich.py efficient_llm/warm_order/loop_screen.py scripts/loop_screen_study.py scripts/dot_sandwich_study.py \
  --module scripts.dot_sandwich_study \
  --handoff $out/dot_fixed_improve_20261125_f01/native-verified.json \
  --handoff-sha256 "$receipt"
/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_compact_t_20261125_f02')
jobs = [(0, 'swst_d3_nt384'), (1, 'swst_d3_nt384'), (0, 'swst_m4d5_nt384'), (1, 'swst_m4d5_nt384'),
        (0, 'swst_d3_nt256'), (1, 'swst_d3_nt256'), (0, 'swst_m4d5_nt256'), (0, 'swst_d3_h4')]
paths = {f's{s}/{a}': base/f's{s}'/a/'progress.json' for s, a in jobs}
for _ in range(36):
    if (base/'failure.json').is_file():
        break
    if len([p for p in paths.values() if p.is_file()]) == len(paths):
        break
    time.sleep(5)
print('F02_FAILURE', (base/'failure.json').read_text()[-3000:] if (base/'failure.json').is_file() else 'none', flush=True)
print('F02_PROGRESS', json.dumps({a: json.loads(p.read_text()) for a, p in paths.items() if p.is_file()}), flush=True)
print('F02_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
