#1 +240+a
#th2-ohmv-dot-vs-loops-2x-f13-20261126
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
date -u
out=/mnt/local/_outputs/deep-llms_th2
if [ -e "$out/dot_vs_loops_2x_20261126_f13" ]; then echo F13_ALREADY_LAUNCHED; cat $out/dot_vs_loops_2x_20261126_f13/status.json || true; exit 0; fi
/usr/bin/python3 - <<'PY'
import json
from pathlib import Path
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_vs_dense8_2x_20261126_f12')
t = json.loads((base/'terminal.json').read_text())
print('F12_TERMINAL', json.dumps(t), flush=True)
assert t['native_verified'] is True, 'native not verified'
PY
receipt=$(sha256sum $out/dot_vs_dense8_2x_20261126_f12/native-verified.json | cut -d' ' -f1)
echo "NATIVE_RECEIPT $receipt"
[ "$receipt" = 6f65ffc39cad1bf3370efa11f54444b21231448c588da00d052bd7eb82d3be6d ]
prior=5aeb88d786915216eb5443fb53ab38756d96ea1339088f78475f5f0a77a783a1
pkg="$PWD/resources/dot_f13_20261126.tar.xz"
echo "f5a2317c05d7c54592d182bdae70a57f009dd8dce9efeea4e78265a1b3fefb47  $pkg" | sha256sum -c -
extra=$(mktemp -d $out/dot_f13_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
export PYTHONPATH=$out/dot_vs_dense8_2x_20261126_f12/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/dot_vs_loops_2x_20261126_f13 \
  --old $out/dot_vs_dense8_2x_20261126_f12/execution --prior-sha256 "$prior" \
  --added efficient_llm/warm_order/loop_screen.py scripts/dot_sandwich_study.py \
  --replaced efficient_llm/warm_order/loop_screen.py scripts/dot_sandwich_study.py \
  --module scripts.dot_sandwich_study \
  --handoff $out/dot_vs_dense8_2x_20261126_f12/native-verified.json \
  --handoff-sha256 "$receipt"
/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_vs_loops_2x_20261126_f13')
jobs = [(0, 'loop_cycle_a8_fa_2x'), (1, 'loop_cycle_a8_fa_2x'), (0, 'loop_cycle_a8_lr14_fa_2x'), (1, 'loop_cycle_a8_lr14_fa_2x'),
        (0, 'loop_prc_a8_fa_2x'), (1, 'loop_prc_a8_fa_2x'), (0, 'dense4_fa_2x'), (1, 'dense4_fa_2x')]
paths = {f's{s}/{a}': base/f's{s}'/a/'progress.json' for s, a in jobs}
for _ in range(36):
    if (base/'failure.json').is_file():
        break
    if len([p for p in paths.values() if p.is_file()]) == len(paths):
        break
    time.sleep(5)
print('F13_FAILURE', (base/'failure.json').read_text()[-3000:] if (base/'failure.json').is_file() else 'none', flush=True)
print('F13_PROGRESS', json.dumps({a: json.loads(p.read_text()) for a, p in paths.items() if p.is_file()}), flush=True)
print('F13_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
