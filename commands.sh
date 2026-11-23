#1 +240+a
#th2-ohmv-wait-a06-then-dot-synth-s01-20261124
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
date -u
out=/mnt/local/_outputs/deep-llms_th2
if [ -e "$out/dot_synth_20261124_s01" ]; then echo SYNTH_ALREADY_LAUNCHED; cat $out/dot_synth_20261124_s01/status.json || true; exit 0; fi
# 1) No signal: wait for a06 (train + downstream) to end on its own; print its exports and failures.
/usr/bin/python3 - <<'PY'
import json, time
from pathlib import Path
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_stop_20261124_a06')
for i in range(900):  # up to 75 min
    if (base/'terminal.json').exists():
        break
    time.sleep(5)
print('A06_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
t = json.loads((base/'terminal.json').read_text())
print('A06_TERMINAL', json.dumps(t), flush=True)
for n in ('failure.json', 'downstream_failure.json'):
    print('A06_' + n, (base/n).read_text()[-1500:] if (base/n).exists() else 'none', flush=True)
for m in sorted((base/'export').glob('*manifest.json')):
    print('A06_EXPORT', m.name, {k: v for k, v in json.loads(m.read_text()).items() if k != 'members'}, flush=True)
assert t['native_verified'] is True, 'native not verified'
PY
receipt=$(sha256sum $out/dot_stop_20261124_a06/native-verified.json | cut -d' ' -f1)
echo "NATIVE_RECEIPT $receipt"
# 2) Launch the synthetic DOT benchmark s01 (old source = a06 execution, pinned hash).
pkg="$PWD/resources/dot_synth_s01_20261124.tar.xz"
echo "fb2113c7c3f0383c56fc3dae58cb8c26920af88658b558f1bdfbf994fd69c11c  $pkg" | sha256sum -c -
extra=$(mktemp -d $out/dot_synth_s01_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
export PYTHONPATH=$out/dot_stop_20261124_a06/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/dot_synth_20261124_s01 \
  --old $out/dot_stop_20261124_a06/execution --prior-sha256 c82f6296afaec820f14cd0918dd422bbe2d62b3de5a88af8be24bb21862e57f1 \
  --added efficient_llm/warm_order/dot_synth.py efficient_llm/warm_order/dot_stop_sandwich.py efficient_llm/warm_order/dot_sandwich.py efficient_llm/warm_order/loop_screen.py scripts/dot_synth_study.py \
  --replaced efficient_llm/warm_order/dot_stop_sandwich.py efficient_llm/warm_order/dot_sandwich.py efficient_llm/warm_order/loop_screen.py \
  --module scripts.dot_synth_study \
  --handoff $out/dot_stop_20261124_a06/native-verified.json \
  --handoff-sha256 "$receipt"
/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_synth_20261124_s01')
arms = ['dense4', 'dense8', 'syn_fixed_d4', 'syn_fixed_d8', 'syn_route_d8', 'syn_stop_d8', 'syn_stopfix_d8', 'syn_budget_d8']
paths = {a: base/'s0'/a/'chain'/'progress.json' for a in arms}
for _ in range(90):
    if (base/'failure.json').is_file():
        print('SYNTH_FAILURE', (base/'failure.json').read_text()[-3000:], flush=True)
        break
    prog = {a: json.loads(p.read_text()) for a, p in paths.items() if p.is_file()}
    if len(prog) == len(paths):
        print('SYNTH_PROGRESS', json.dumps(prog), flush=True)
        break
    time.sleep(5)
print('SYNTH_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
for a in arms:
    log = base/f'train-s0-{a}.log'
    if log.is_file():
        print('SYNTH_LOG', a, log.read_text()[-400:], flush=True)
PROGRESS
