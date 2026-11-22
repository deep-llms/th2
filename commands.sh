#1 +240+a
#th2-ohmv-loop-screen-20261123-a01
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
pkg="$PWD/resources/loop_screen_20261123.tar.xz"
echo "7a30fd4050b995f1d00cb03f6414738d891f90f41632cb9d843bf850907d3696  $pkg" | sha256sum -c -
extra=$(mktemp -d /mnt/local/_outputs/deep-llms_th2/loop_screen_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
export PYTHONPATH=/mnt/local/_outputs/deep-llms_th2/anchor_loop_20261123_a01/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --handoff /mnt/local/_outputs/deep-llms_th2/anchor_loop_20261123_a01/native-verified.json --handoff-sha256 cbcc4c70b3bd308abd5ef13321343e450114aeb41d6afa5224b51d14de3930da

/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/loop_screen_20261123_a01')
arms = ('dense4','dense4_2x','dense8','serial','memory','gated','gated_var','attnres')
for _ in range(40):
    if (base/'failure.json').is_file():
        print('LOOP_SCREEN_FAILURE', (base/'failure.json').read_text()[-2000:], flush=True)
        break
    prog = {a: json.loads(p.read_text()) for a in arms if (p := base/'s0'/a/'progress.json').is_file()}
    if len(prog) == len(arms):
        print('LOOP_SCREEN_PROGRESS', json.dumps(prog), flush=True)
        break
    time.sleep(5)
status = base/'status.json'
print('LOOP_SCREEN_STATUS', status.read_text() if status.is_file() else 'absent', flush=True)
PROGRESS
