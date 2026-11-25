#1 +240+a
#th2-ohmv-dot-tmlp-f04-20261125
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
date -u
out=/mnt/local/_outputs/deep-llms_th2
if [ -e "$out/dot_tmlp_20261125_f04" ]; then echo F04_ALREADY_LAUNCHED; cat $out/dot_tmlp_20261125_f04/status.json || true; exit 0; fi
/usr/bin/python3 - <<'PY'
import json
from pathlib import Path
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_shared_heads_20261125_f03')
t = json.loads((base/'terminal.json').read_text())
print('F03_TERMINAL', json.dumps(t), flush=True)
assert t['native_verified'] is True, 'native not verified'
PY
receipt=$(sha256sum $out/dot_shared_heads_20261125_f03/native-verified.json | cut -d' ' -f1)
echo "NATIVE_RECEIPT $receipt"
[ "$receipt" = c4d6d61874c8cd6976582fcf80496157d17cba7ea5cdcabcc11bf215676a4961 ]
prior=7bec6e0094e7bff7854c1247d604092b3bb0377b5afb2d8e0f0d391a05d5efa0
pkg="$PWD/resources/dot_f04_20261125.tar.xz"
echo "ce21969fef41002868ad37dd277f39f38f9a3861a148c4777b7a455b85d9f9fd  $pkg" | sha256sum -c -
extra=$(mktemp -d $out/dot_f04_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
export PYTHONPATH=$out/dot_shared_heads_20261125_f03/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/dot_tmlp_20261125_f04 \
  --old $out/dot_shared_heads_20261125_f03/execution --prior-sha256 "$prior" \
  --added efficient_llm/warm_order/dot_stop_sandwich.py efficient_llm/warm_order/loop_baselines.py efficient_llm/warm_order/loop_screen.py scripts/dot_sandwich_study.py \
  --replaced efficient_llm/warm_order/dot_stop_sandwich.py efficient_llm/warm_order/loop_baselines.py efficient_llm/warm_order/loop_screen.py scripts/dot_sandwich_study.py \
  --module scripts.dot_sandwich_study \
  --handoff $out/dot_shared_heads_20261125_f03/native-verified.json \
  --handoff-sha256 "$receipt"
/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_tmlp_20261125_f04')
jobs = [(1, 'swst_fixed_d3_tmlp'), (1, 'swst_d3_h4'), (0, 'swst_d4_tmlp'), (1, 'swst_d4_tmlp'),
        (0, 'loop_cycle_a8'), (1, 'loop_cycle_a8'), (0, 'swst_d3_tmlp_film'), (1, 'swst_d3_tmlp_film')]
paths = {f's{s}/{a}': base/f's{s}'/a/'progress.json' for s, a in jobs}
for _ in range(36):
    if (base/'failure.json').is_file():
        break
    if len([p for p in paths.values() if p.is_file()]) == len(paths):
        break
    time.sleep(5)
print('F04_FAILURE', (base/'failure.json').read_text()[-3000:] if (base/'failure.json').is_file() else 'none', flush=True)
print('F04_PROGRESS', json.dumps({a: json.loads(p.read_text()) for a, p in paths.items() if p.is_file()}), flush=True)
print('F04_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
