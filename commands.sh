#1 +240+a
#th2-ohmv-loop-stop-20261123-a01
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
pkg="$PWD/resources/loop_stop_20261123.tar.xz"
echo "53307489f576112731948aa284517d7eb47656411fd9f43d142a0f381447d07a  $pkg" | sha256sum -c -
extra=$(mktemp -d /mnt/local/_outputs/deep-llms_th2/loop_stop_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
out=/mnt/local/_outputs/deep-llms_th2
export PYTHONPATH=$out/loop_screen2_20261123_a02/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/loop_stop_20261123_a01 \
  --old $out/loop_screen2_20261123_a02/execution --prior-sha256 b4f1569e47ea47e45e4ec254bb87457739388a13adb464a2f5e34f16b4183344 \
  --added efficient_llm/warm_order/loop_stop.py scripts/loop_stop_study.py \
  --module scripts.loop_stop_study \
  --handoff $out/loop_screen2_20261123_a02/native-verified.json \
  --handoff-sha256 c9069c1464411c09ec5638f74bf37277bb1e3e48efdf021bd1b1da17c0b378c5

/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/loop_stop_20261123_a01')
for _ in range(40):
    if (base/'failure.json').is_file():
        print('LOOP_STOP_FAILURE', (base/'failure.json').read_text()[-2000:], flush=True)
        break
    st = base/'status.json'
    if st.is_file() and json.loads(st.read_text()).get('stage') == 'stop':
        break
    time.sleep(5)
print('LOOP_STOP_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
