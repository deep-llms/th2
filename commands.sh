#1 +240+a
#th2-ohmv-moe-dyn-20261123-a02
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
pkg="$PWD/resources/moe_dyn_a02_20261123.tar.xz"
echo "005e3910f8dc1934b0b1967f8086b49d98b15488b093c4d9d00a46d763265bfb  $pkg" | sha256sum -c -
extra=$(mktemp -d /mnt/local/_outputs/deep-llms_th2/moe_dyn_a02_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
out=/mnt/local/_outputs/deep-llms_th2
export PYTHONPATH=$out/moe_dyn_20261123_a01/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/moe_dyn_20261123_a02 \
  --old $out/moe_dyn_20261123_a01/execution --prior-sha256 e8f7d09f80d1c4e7b2c500de6d976f302a2223d106154732325d3728d87b31d0 \
  --added scripts/loop_screen_study.py scripts/moe_dyn_study.py \
  --replaced scripts/loop_screen_study.py scripts/moe_dyn_study.py \
  --module scripts.moe_dyn_study \
  --handoff $out/moe_dyn_20261123_a01/native-verified.json \
  --handoff-sha256 6974335ce7897489ff31059452a8224cad24815613649097cf748088d0b4b777

/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/moe_dyn_20261123_a02')
paths = {a: base/'moe'/a/'progress.json' for a in ('moe2_top1', 'moe4_top1', 'moe8_top2')}
paths.update({f's{s}/{a}': base/'loop'/f's{s}'/a/'progress.json' for s, a in ((0, 'dynattn'), (0, 'dynattn8'), (0, 'loop_dynattn'), (1, 'dynattn'), (1, 'loop_dynattn'))})
for _ in range(44):
    if (base/'failure.json').is_file():
        print('MOE_DYN_FAILURE', (base/'failure.json').read_text()[-2000:], flush=True)
        break
    prog = {a: json.loads(p.read_text()) for a, p in paths.items() if p.is_file()}
    if len(prog) == len(paths):
        print('MOE_DYN_PROGRESS', json.dumps(prog), flush=True)
        break
    time.sleep(5)
print('MOE_DYN_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
