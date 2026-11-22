#1 +240+a
#th2-ohmv-loop-screen4-20261123-a02
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
pkg="$PWD/resources/loop_screen4_20261123.tar.xz"
echo "4a485b47ca31efa64313fcfe32772f937361752507771d2177069bf28362f27f  $pkg" | sha256sum -c -
extra=$(mktemp -d /mnt/local/_outputs/deep-llms_th2/loop_screen4_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
out=/mnt/local/_outputs/deep-llms_th2
export PYTHONPATH=$out/loop_stop_20261123_a01/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/loop_screen4_20261123_a02 \
  --old $out/loop_stop_20261123_a01/execution --prior-sha256 b771fe99776ad9ae79c97462e0c674b9f185d122d83687a83ae7c2398a7448cf \
  --added efficient_llm/warm_order/loop_screen.py scripts/loop_screen_study.py \
  --replaced efficient_llm/warm_order/loop_screen.py scripts/loop_screen_study.py \
  --module scripts.loop_screen_study \
  --handoff $out/loop_screen4_20261123_a01/native-verified.json \
  --handoff-sha256 ba1d26b31f87097861e1088c20d872061b300416c4546173053d94f6a4374d2b

/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/loop_screen4_20261123_a02')
jobs = [(0,'loop_attnres_tokvar'),(0,'gated_tokvar'),(1,'loop_attnres'),(1,'attnres'),(1,'dense4'),(1,'attnres8'),(0,'loop_attnres_2x'),(0,'dense8_2x')]
for _ in range(40):
    if (base/'failure.json').is_file():
        print('LOOP_SCREEN4_FAILURE', (base/'failure.json').read_text()[-2000:], flush=True)
        break
    prog = {f's{s}/{a}': json.loads(p.read_text()) for s, a in jobs if (p := base/f's{s}'/a/'progress.json').is_file()}
    if len(prog) == len(jobs):
        print('LOOP_SCREEN4_PROGRESS', json.dumps(prog), flush=True)
        break
    time.sleep(5)
print('LOOP_SCREEN4_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
