#1 +240+a
#th2-ohmv-loop-screen-downstream-20261123-a01
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
pkg="$PWD/resources/loop_downstream_20261123.tar.xz"
echo "ba8cb1fed599e291b446df4f78c77f47df49d403750ff48a1ba297b894a6b7c9  $pkg" | sha256sum -c -
extra=$(mktemp -d /mnt/local/_outputs/deep-llms_th2/loop_downstream_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
screen=/mnt/local/_outputs/deep-llms_th2/loop_screen_20261123_a01
export PYTHONPATH=$screen/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base /mnt/local/_outputs/deep-llms_th2/loop_screen_downstream_20261123_a01 \
  --old $screen/execution --prior-sha256 671dd1883eed7e9366190e5e1fee923896ef6b9efa4c885fe07dba20c7a0facd \
  --added efficient_llm/warm_order/loop_screen_downstream.py scripts/loop_screen_downstream_study.py \
  --module scripts.loop_screen_downstream_study \
  --handoff $screen/native-verified.json --handoff-sha256 2469311b64986984b38668b1433f4ac1a31a6b51a9567c166a1f0cc5aeb767d4

/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/loop_screen_downstream_20261123_a01')
for _ in range(40):
    if (base/'failure.json').is_file():
        print('DOWNSTREAM_FAILURE', (base/'failure.json').read_text()[-2000:], flush=True)
        break
    st = base/'status.json'
    if st.is_file() and json.loads(st.read_text()).get('stage') == 'score':
        break
    time.sleep(5)
print('DOWNSTREAM_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
