#1 +240+a
#th2-ohmv-loop-screen2-20261123-a01
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
pkg="$PWD/resources/loop_screen2_20261123.tar.xz"
echo "d4fd47c9d2cdf5ac2ff5d679e12c7ae3970e007832114a0c70ebff39113dea38  $pkg" | sha256sum -c -
extra=$(mktemp -d /mnt/local/_outputs/deep-llms_th2/loop_screen2_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
out=/mnt/local/_outputs/deep-llms_th2
export PYTHONPATH=$out/loop_screen_20261123_a01/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/loop_screen2_20261123_a01 \
  --old $out/loop_screen_20261123_a01/execution --prior-sha256 671dd1883eed7e9366190e5e1fee923896ef6b9efa4c885fe07dba20c7a0facd \
  --added efficient_llm/warm_order/loop_screen.py scripts/loop_screen_study.py \
  --replaced efficient_llm/warm_order/loop_screen.py scripts/loop_screen_study.py \
  --module scripts.loop_screen_study \
  --handoff $out/loop_screen_downstream_20261123_a01/native-verified.json \
  --handoff-sha256 880ed0dd713d8e55cb8b14813e6d676269e3001ace31ff3773343ef1f39f6ac6

/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/loop_screen2_20261123_a01')
jobs = [(1,'memory'),(1,'dense8'),(0,'memory_var'),(0,'loop_attnres'),(0,'loop_attnres_var'),(0,'attnres8'),(0,'attnres_2x'),(0,'memory_2x')]
for _ in range(40):
    if (base/'failure.json').is_file():
        print('LOOP_SCREEN2_FAILURE', (base/'failure.json').read_text()[-2000:], flush=True)
        break
    prog = {f's{s}/{a}': json.loads(p.read_text()) for s, a in jobs if (p := base/f's{s}'/a/'progress.json').is_file()}
    if len(prog) == len(jobs):
        print('LOOP_SCREEN2_PROGRESS', json.dumps(prog), flush=True)
        break
    time.sleep(5)
print('LOOP_SCREEN2_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
