#1 +240+a
#th2-ohmv-dot-synth-s09-20261124
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
date -u
out=/mnt/local/_outputs/deep-llms_th2
if [ -e "$out/dot_synth_20261124_s09" ]; then echo S09_ALREADY_LAUNCHED; cat $out/dot_synth_20261124_s09/status.json || true; exit 0; fi
# 1) s08 has ended; confirm its terminal (results already read).
/usr/bin/python3 - <<'PY'
import json, time
from pathlib import Path
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_synth_20261124_s08')
for i in range(60):  # s08 already ended; short guard
    if (base/'terminal.json').exists() or (base/'failure.json').exists():
        break
    time.sleep(5)
print('S08_FAILURE', (base/'failure.json').read_text()[-1500:] if (base/'failure.json').exists() else 'none', flush=True)
t = json.loads((base/'terminal.json').read_text())
print('S08_TERMINAL', json.dumps(t), flush=True)
m = base/'export'/'manifest.json'
print('S08_EXPORT', {k: v for k, v in json.loads(m.read_text()).items() if k != 'members'} if m.exists() else 'absent', flush=True)
assert t['native_verified'] is True, 'native not verified'
PY
receipt=$(sha256sum $out/dot_synth_20261124_s08/native-verified.json | cut -d' ' -f1)
echo "NATIVE_RECEIPT $receipt"
# 2) Launch s09 (pure-loop control on chain; greedy-body prior router on opseq); old source = s08 execution.
pkg="$PWD/resources/dot_synth_s09_20261124.tar.xz"
echo "f7fbd6f9cf62b32fc27186453cbea3909510365543109219414e2b8ad5e2ec32  $pkg" | sha256sum -c -
extra=$(mktemp -d $out/dot_synth_s09_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
export PYTHONPATH=$out/dot_synth_20261124_s08/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/dot_synth_20261124_s09 \
  --old $out/dot_synth_20261124_s08/execution --prior-sha256 7436e2dc4cdddb0f968f42c210a23572d8c61bf6e4273d890cad9053d66f48c9 \
  --added efficient_llm/warm_order/dot_stop_sandwich.py efficient_llm/warm_order/loop_screen.py scripts/dot_synth_study.py \
  --replaced efficient_llm/warm_order/dot_stop_sandwich.py efficient_llm/warm_order/loop_screen.py scripts/dot_synth_study.py \
  --module scripts.dot_synth_study \
  --handoff $out/dot_synth_20261124_s08/native-verified.json \
  --handoff-sha256 "$receipt"
/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_synth_20261124_s09')
for _ in range(24):
    if (base/'failure.json').is_file() or (base/'status.json').is_file() and 'train' in (base/'status.json').read_text():
        break
    time.sleep(5)
print('S09_FAILURE', (base/'failure.json').read_text()[-3000:] if (base/'failure.json').is_file() else 'none', flush=True)
print('S09_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
