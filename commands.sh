#1 +240+a
#th2-ohmv-dot-synth-s11-20261125
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
date -u
out=/mnt/local/_outputs/deep-llms_th2
if [ -e "$out/dot_synth_20261125_s11" ]; then echo S11_ALREADY_LAUNCHED; cat $out/dot_synth_20261125_s11/status.json || true; exit 0; fi
# 1) s10 has ended; confirm its terminal (results already read).
/usr/bin/python3 - <<'PY'
import json, time
from pathlib import Path
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_synth_20261125_s10')
for i in range(60):  # s10 already ended; short guard
    if (base/'terminal.json').exists() or (base/'failure.json').exists():
        break
    time.sleep(5)
print('S10_FAILURE', (base/'failure.json').read_text()[-1500:] if (base/'failure.json').exists() else 'none', flush=True)
t = json.loads((base/'terminal.json').read_text())
print('S10_TERMINAL', json.dumps(t), flush=True)
m = base/'export'/'manifest.json'
print('S10_EXPORT', {k: v for k, v in json.loads(m.read_text()).items() if k != 'members'} if m.exists() else 'absent', flush=True)
assert t['native_verified'] is True, 'native not verified'
PY
receipt=$(sha256sum $out/dot_synth_20261125_s10/native-verified.json | cut -d' ' -f1)
echo "NATIVE_RECEIPT $receipt"
# 2) Launch s11 (4-seed completion under random depth, router without prior); old source = s10 execution.
pkg="$PWD/resources/dot_synth_s11_20261125.tar.xz"
echo "291cd3afd368678eb71963253134ec17ee522112df02ad2ef624b60f677a324f  $pkg" | sha256sum -c -
extra=$(mktemp -d $out/dot_synth_s11_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
export PYTHONPATH=$out/dot_synth_20261125_s10/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/dot_synth_20261125_s11 \
  --old $out/dot_synth_20261125_s10/execution --prior-sha256 f9b35cdf3ebb60711ce3c3b58955af60a58e5ee632411bd5851c8c170acb1713 \
  --added efficient_llm/warm_order/loop_screen.py scripts/dot_synth_study.py \
  --replaced efficient_llm/warm_order/loop_screen.py scripts/dot_synth_study.py \
  --module scripts.dot_synth_study \
  --handoff $out/dot_synth_20261125_s10/native-verified.json \
  --handoff-sha256 "$receipt"
/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_synth_20261125_s11')
for _ in range(24):
    if (base/'failure.json').is_file() or (base/'status.json').is_file() and 'train' in (base/'status.json').read_text():
        break
    time.sleep(5)
print('S11_FAILURE', (base/'failure.json').read_text()[-3000:] if (base/'failure.json').is_file() else 'none', flush=True)
print('S11_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
