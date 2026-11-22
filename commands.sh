#1 +240+a
#th2-ohmv-loop-screen2-20261123-a02
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
pkg="$PWD/resources/loop_screen2_20261123.tar.xz"
echo "39c46995d7d83494ba2242a1bfd28fb627b616a8ded4dd19246e8b11668a507d  $pkg" | sha256sum -c -
extra=$(mktemp -d /mnt/local/_outputs/deep-llms_th2/loop_screen2_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
out=/mnt/local/_outputs/deep-llms_th2
export PYTHONPATH=$out/loop_screen_20261123_a01/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/loop_screen2_20261123_a02 \
  --old $out/loop_screen_20261123_a01/execution --prior-sha256 671dd1883eed7e9366190e5e1fee923896ef6b9efa4c885fe07dba20c7a0facd \
  --added efficient_llm/warm_order/loop_screen.py scripts/loop_screen_study.py \
  --replaced efficient_llm/warm_order/loop_screen.py scripts/loop_screen_study.py \
  --module scripts.loop_screen_study \
  --handoff $out/loop_screen2_20261123_a01/native-verified.json \
  --handoff-sha256 521582e93b7b1fe215ce64dacc49cab0330585c109677edbae224203f4e3e30a

/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/loop_screen2_20261123_a02')
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
