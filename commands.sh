#1 +240+a
#th2-ohmv-loop-screen4-downstream-20261123-a01
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
pkg="$PWD/resources/loop_downstream4_20261123.tar.xz"
echo "a3f26c6683a6ab4c996ecfcf46c58a08f0b09b53124f1c6e732ca1f37a8f0e30  $pkg" | sha256sum -c -
extra=$(mktemp -d /mnt/local/_outputs/deep-llms_th2/loop_downstream4_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
out=/mnt/local/_outputs/deep-llms_th2
export PYTHONPATH=$out/loop_stop2_20261123_a01/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/loop_screen4_downstream_20261123_a01 \
  --old $out/loop_stop2_20261123_a01/execution --prior-sha256 2dbca58304be81cfeab67ec3da162de5e6b2b058e1450fe93e6af0e0678e5c42 \
  --added efficient_llm/warm_order/loop_screen_downstream.py scripts/loop_screen_downstream_study.py \
  --module scripts.loop_screen_downstream_study \
  --handoff $out/loop_stop2_20261123_a01/native-verified.json \
  --handoff-sha256 ff346552c60b1f029cfb3ca6d3268f5b620f6e598a864ec3a8b96b5c2278714e

/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/loop_screen4_downstream_20261123_a01')
for _ in range(40):
    if (base/'failure.json').is_file():
        print('DOWNSTREAM2_FAILURE', (base/'failure.json').read_text()[-2000:], flush=True)
        break
    st = base/'status.json'
    if st.is_file() and json.loads(st.read_text()).get('stage') == 'score':
        break
    time.sleep(5)
print('DOWNSTREAM2_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
