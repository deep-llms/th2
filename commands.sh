#1 +240+a
#th2-ohmv-scale-downstream-20261123-a01
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
pkg="$PWD/resources/scale_downstream_20261123.tar.xz"
echo "98356cd68093f210a89aacbb9d84ce95e1eec40f71bc692dcf8eb48e33f7acf2  $pkg" | sha256sum -c -
extra=$(mktemp -d /mnt/local/_outputs/deep-llms_th2/scale_downstream_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
out=/mnt/local/_outputs/deep-llms_th2
export PYTHONPATH=$out/scale_round_20261123_a01/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/scale_downstream_20261123_a01 \
  --old $out/scale_round_20261123_a01/execution --prior-sha256 bda16689664a51fc9cf3fef82f20f5731e81c55632e50adf143ec51011efec6d \
  --added efficient_llm/warm_order/loop_screen_downstream.py scripts/loop_screen_downstream_study.py \
  --module scripts.loop_screen_downstream_study \
  --handoff $out/scale_round_20261123_a01/native-verified.json \
  --handoff-sha256 fbc40ff7f6cc56e540737df28d2811f4fad8e082053128c733f7e65ee09c0f8f

/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/scale_downstream_20261123_a01')
for _ in range(40):
    if (base/'failure.json').is_file():
        print('SCALE_DS_FAILURE', (base/'failure.json').read_text()[-2000:], flush=True)
        break
    st = base/'status.json'
    if st.is_file() and json.loads(st.read_text()).get('stage') == 'score':
        break
    time.sleep(5)
print('SCALE_DS_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
