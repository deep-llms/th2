#1 +240+a
#th2-ohmv-moe-downstream-20261123-a01
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
pkg="$PWD/resources/moe_downstream_20261123.tar.xz"
echo "7b72e28ce689aeb49c575d31a846c8ebb88e5fb9f149112e3814ca1a70128e48  $pkg" | sha256sum -c -
extra=$(mktemp -d /mnt/local/_outputs/deep-llms_th2/moe_downstream_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
out=/mnt/local/_outputs/deep-llms_th2
export PYTHONPATH=$out/moe_loop_20261123_a01/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/moe_downstream_20261123_a01 \
  --old $out/moe_loop_20261123_a01/execution --prior-sha256 9781a5ac5834099a9c6feaba8a3f27d80c318c7de67af80023b2d8aaacb03408 \
  --added scripts/loop_screen_downstream_study.py \
  --replaced scripts/loop_screen_downstream_study.py \
  --module scripts.loop_screen_downstream_study \
  --handoff $out/moe_loop_20261123_a01/native-verified.json \
  --handoff-sha256 874ae44a70671e4bd2283ee5a9edbf71e06cfdd32735bd618f1abc7f55956a63

/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/moe_downstream_20261123_a01')
for _ in range(40):
    if (base/'failure.json').is_file():
        print('MOE_DS_FAILURE', (base/'failure.json').read_text()[-2000:], flush=True)
        break
    st = base/'status.json'
    if st.is_file() and json.loads(st.read_text()).get('stage') == 'score':
        break
    time.sleep(5)
print('MOE_DS_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
