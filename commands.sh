#1 +240+a
#th2-ohmv-loop-stop2-20261123-a01
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
pkg="$PWD/resources/loop_stop2_20261123.tar.xz"
echo "f2d3f05b24e8b2bf9136f5cb6f0ff17f784d1f0f3a8586dd75d735a395ae1335  $pkg" | sha256sum -c -
extra=$(mktemp -d /mnt/local/_outputs/deep-llms_th2/loop_stop2_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
out=/mnt/local/_outputs/deep-llms_th2
export PYTHONPATH=$out/loop_screen4_20261123_a02/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/loop_stop2_20261123_a01 \
  --old $out/loop_screen4_20261123_a02/execution --prior-sha256 291b293be41b9501c1042123bd82f4f395bdc8442ab1cbea2c55a57f13f7e749 \
  --added scripts/loop_stop_study.py --replaced scripts/loop_stop_study.py \
  --module scripts.loop_stop_study \
  --handoff $out/loop_screen4_20261123_a02/native-verified.json \
  --handoff-sha256 64bb962452dc73d8fb7a79b9cc20cd6a39f7665cb09ca68dce4530354521218c

/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/loop_stop2_20261123_a01')
for _ in range(40):
    if (base/'failure.json').is_file():
        print('LOOP_STOP2_FAILURE', (base/'failure.json').read_text()[-2000:], flush=True)
        break
    st = base/'status.json'
    if st.is_file() and json.loads(st.read_text()).get('stage') == 'stop':
        break
    time.sleep(5)
print('LOOP_STOP2_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
