#1 +240+a
#th2-ohmv-family-downstream-20261123-a01
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
pkg="$PWD/resources/family_downstream_20261123.tar.xz"
echo "eaf7ccb4e223854b6683e3da7f864abe5215257455285635f49c36279c5fea2a  $pkg" | sha256sum -c -
extra=$(mktemp -d /mnt/local/_outputs/deep-llms_th2/family_downstream_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
out=/mnt/local/_outputs/deep-llms_th2
export PYTHONPATH=$out/scale_downstream_20261123_a01/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/family_downstream_20261123_a01 \
  --old $out/scale_downstream_20261123_a01/execution --prior-sha256 27cb6d35ceafc7fc8c78197c7e11b5b8d6db4aaedf6cc3eed8f5c2f32207b5a4 \
  --added scripts/family_downstream_study.py \
  --module scripts.family_downstream_study \
  --handoff $out/scale_downstream_20261123_a01/native-verified.json \
  --handoff-sha256 4b7ec19db4e97ee7be5be9dc70afbd6579c0ca847e614d3947fd2bf3029ed06b

/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/family_downstream_20261123_a01')
for _ in range(40):
    if (base/'failure.json').is_file():
        print('FAMILY_DS_FAILURE', (base/'failure.json').read_text()[-2000:], flush=True)
        break
    st = base/'status.json'
    if st.is_file() and json.loads(st.read_text()).get('stage') == 'score':
        break
    time.sleep(5)
print('FAMILY_DS_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
print('FAMILY_DS_PREPARED', (base/'prepared.json').read_text()[:3000] if (base/'prepared.json').is_file() else 'absent', flush=True)
PROGRESS
