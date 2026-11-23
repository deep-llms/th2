#1 +240+a
#th2-ohmv-moe-loop-20261123-a01
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
pkg="$PWD/resources/moe_loop_20261123.tar.xz"
echo "958775cccea6f74a13dba73e2f21fa2900fe493f8201f4f98983db14c1fd9cfd  $pkg" | sha256sum -c -
extra=$(mktemp -d /mnt/local/_outputs/deep-llms_th2/moe_loop_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
out=/mnt/local/_outputs/deep-llms_th2
export PYTHONPATH=$out/moe_dyn_20261123_a02/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/moe_loop_20261123_a01 \
  --old $out/moe_dyn_20261123_a02/execution --prior-sha256 47327f41b5e623d5172b90bcdfbdf0bc77bcf130ccb64a0849d7daf849d3645f \
  --added efficient_llm/warm_order/loop_screen.py scripts/loop_screen_study.py scripts/moe_loop_study.py \
  --replaced efficient_llm/warm_order/loop_screen.py scripts/loop_screen_study.py \
  --module scripts.moe_loop_study \
  --handoff $out/moe_dyn_20261123_a02/native-verified.json \
  --handoff-sha256 d6a6e5c97ab37ed2fff007bf67cc7e29869b6247a00a734e36f470819123b271

/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/moe_loop_20261123_a01')
jobs = [(0, 'attnres_moe'), (1, 'attnres_moe'), (0, 'loop_attnres_moe'), (1, 'loop_attnres_moe'),
        (0, 'dense4_moe'), (0, 'attnres8_moe'), (0, 'dynattn_tanh'), (1, 'dynattn_tanh')]
paths = {f's{s}/{a}': base/f's{s}'/a/'progress.json' for s, a in jobs}
for _ in range(44):
    if (base/'failure.json').is_file():
        print('MOE_LOOP_FAILURE', (base/'failure.json').read_text()[-2000:], flush=True)
        break
    prog = {a: json.loads(p.read_text()) for a, p in paths.items() if p.is_file()}
    if len(prog) == len(paths):
        print('MOE_LOOP_PROGRESS', json.dumps(prog), flush=True)
        break
    time.sleep(5)
print('MOE_LOOP_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
