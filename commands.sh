#1 +240+a
#th2-ohmv-dot-synth-s02-20261124
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
date -u
out=/mnt/local/_outputs/deep-llms_th2
if [ -e "$out/dot_synth_20261124_s02" ]; then echo S02_ALREADY_LAUNCHED; cat $out/dot_synth_20261124_s02/status.json || true; exit 0; fi
/usr/bin/python3 - <<'PY'
import json
from pathlib import Path
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_synth_20261124_s01')
t = json.loads((base/'terminal.json').read_text())
print('S01_TERMINAL', json.dumps(t), flush=True)
assert t['native_verified'] is True, 'native not verified'
PY
receipt=$(sha256sum $out/dot_synth_20261124_s01/native-verified.json | cut -d' ' -f1)
echo "NATIVE_RECEIPT $receipt"
pkg="$PWD/resources/dot_synth_s02_20261124.tar.xz"
echo "75b672e62ad8fdb530e1f4285a6ef2ffe26a391ea5a69e04077660aa2173e736  $pkg" | sha256sum -c -
extra=$(mktemp -d $out/dot_synth_s02_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
export PYTHONPATH=$out/dot_synth_20261124_s01/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/dot_synth_20261124_s02 \
  --old $out/dot_synth_20261124_s01/execution --prior-sha256 99271d05fed2b6664f2e2095a7ec7f4884efabfd0cdfd44ceec798269a0cf164 \
  --added efficient_llm/warm_order/dot_synth.py efficient_llm/warm_order/dot_stop_sandwich.py efficient_llm/warm_order/loop_screen.py scripts/dot_synth_study.py \
  --replaced efficient_llm/warm_order/dot_synth.py efficient_llm/warm_order/dot_stop_sandwich.py efficient_llm/warm_order/loop_screen.py scripts/dot_synth_study.py \
  --module scripts.dot_synth_study \
  --handoff $out/dot_synth_20261124_s01/native-verified.json \
  --handoff-sha256 "$receipt"
/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_synth_20261124_s02')
arms = ['dense4', 'dense8', 'syn_fixed_d4', 'syn_fixed_d8', 'syn_route_d8', 'syn_stop_d8w', 'syn_stopfix_d8w', 'syn_budget_d8w']
paths = {a: base/'s0'/a/'chain'/'progress.json' for a in arms}
for _ in range(36):
    if (base/'failure.json').is_file():
        print('S02_FAILURE', (base/'failure.json').read_text()[-3000:], flush=True)
        break
    prog = {a: json.loads(p.read_text()) for a, p in paths.items() if p.is_file()}
    if len(prog) == len(paths):
        break
    time.sleep(5)
print('S02_PROGRESS', json.dumps({a: json.loads(p.read_text()) for a, p in paths.items() if p.is_file()}), flush=True)
print('S02_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
for a in arms:
    log = base/f'train-s0-{a}.log'
    if log.is_file() and ('Error' in log.read_text() or 'Traceback' in log.read_text()):
        print('S02_ERR', a, log.read_text()[-1500:], flush=True)
PROGRESS
