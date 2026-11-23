#1 +240+a
#th2-ohmv-moe-scale-20261123-a01
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
pkg="$PWD/resources/moe_scale_20261123.tar.xz"
echo "5a182dc96c9359450263e2fd5fc2fa40ff88f25caa2c0da71350218ab7b805f7  $pkg" | sha256sum -c -
extra=$(mktemp -d /mnt/local/_outputs/deep-llms_th2/moe_scale_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
out=/mnt/local/_outputs/deep-llms_th2
export PYTHONPATH=$out/moe_variants_20261123_a01/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/moe_scale_20261123_a01 \
  --old $out/moe_variants_20261123_a01/execution --prior-sha256 2db9f7e29d21cd7619ca5cc45312f9a3befb5013c30e86642c578170385cdd35 \
  --added efficient_llm/warm_order/loop_screen.py scripts/moe_scale_study.py \
  --replaced efficient_llm/warm_order/loop_screen.py \
  --module scripts.moe_scale_study \
  --handoff $out/moe_variants_20261123_a01/native-verified.json \
  --handoff-sha256 71326f67f0ed9dc420e01c5e97b1f9dc35e453f82509b7015d7e534f03987c92

/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/moe_scale_20261123_a01')
jobs = [(0, 'attnres_moe_4x'), (0, 'loop_attnres_moe_4x'), (0, 'attnres8_moe_4x'), (1, 'attnres_moe_4x'),
        (1, 'loop_attnres_moe_4x'), (1, 'attnres_4x'), (1, 'loop_attnres_4x'), (1, 'attnres8_4x')]
paths = {f's{s}/{a}': base/f's{s}'/a/'progress.json' for s, a in jobs}
for _ in range(44):
    if (base/'failure.json').is_file():
        print('MOE_SCALE_FAILURE', (base/'failure.json').read_text()[-2000:], flush=True)
        break
    prog = {a: json.loads(p.read_text()) for a, p in paths.items() if p.is_file()}
    if len(prog) == len(paths):
        print('MOE_SCALE_PROGRESS', json.dumps(prog), flush=True)
        break
    time.sleep(5)
print('MOE_SCALE_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
