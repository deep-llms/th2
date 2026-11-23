#1 +240+a
#th2-ohmv-scale-round-20261123-a01
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
pkg="$PWD/resources/scale_20261123.tar.xz"
echo "8988dda72b9ba0f0b37157fd5e902bf1af097f44eb9a0e2dfd7171ea0f324d05  $pkg" | sha256sum -c -
extra=$(mktemp -d /mnt/local/_outputs/deep-llms_th2/scale_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
out=/mnt/local/_outputs/deep-llms_th2
export PYTHONPATH=$out/domain_screen_20261123_a01/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/scale_round_20261123_a01 \
  --old $out/domain_screen_20261123_a01/execution --prior-sha256 6e5e36350965f638d0c3a9895e8639aa6799010e82a74c0369947d3a27e80a83 \
  --added efficient_llm/warm_order/loop_screen.py scripts/loop_screen_study.py scripts/domain_screen_study.py scripts/scale_round_study.py \
  --replaced efficient_llm/warm_order/loop_screen.py scripts/loop_screen_study.py scripts/domain_screen_study.py \
  --module scripts.scale_round_study \
  --handoff $out/domain_screen_20261123_a01/native-verified.json \
  --handoff-sha256 8dd71a0fdd9013a8bad1949b81adc28d41a1c439717f52df5c4f84e1ff395841

/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/scale_round_20261123_a01')
paths = {a: base/'loop'/'s0'/a/'progress.json' for a in ('loop_attnres_4x', 'attnres_4x', 'attnres8_4x', 'dense8_4x', 'dense4_4x')}
paths.update({'domain_'+a: base/'domain'/a/'progress.json' for a in ('single', 'oracle')})
for _ in range(44):
    if (base/'failure.json').is_file():
        print('SCALE_FAILURE', (base/'failure.json').read_text()[-2000:], flush=True)
        break
    prog = {a: json.loads(p.read_text()) for a, p in paths.items() if p.is_file()}
    if len(prog) == len(paths):
        print('SCALE_PROGRESS', json.dumps(prog), flush=True)
        break
    time.sleep(5)
print('SCALE_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
