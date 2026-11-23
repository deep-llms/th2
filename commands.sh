#1 +240+a
#th2-ohmv-moe-dyn-20261123-a01
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
pkg="$PWD/resources/moe_dyn_20261123.tar.xz"
echo "6c5fe545622facbe7a88b3f5993b133f23c2141a7264fd7e1119d369d887320a  $pkg" | sha256sum -c -
extra=$(mktemp -d /mnt/local/_outputs/deep-llms_th2/moe_dyn_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
out=/mnt/local/_outputs/deep-llms_th2
export PYTHONPATH=$out/router_screen_20261123_a01/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/moe_dyn_20261123_a01 \
  --old $out/router_screen_20261123_a01/execution --prior-sha256 83d4e9e48590c6a152a1b02077733c185ee6ae378470e72ee27f27483a17dc5e \
  --added efficient_llm/warm_order/loop_screen.py efficient_llm/warm_order/router_screen.py scripts/loop_screen_study.py scripts/router_screen_study.py scripts/moe_dyn_study.py \
  --replaced efficient_llm/warm_order/loop_screen.py efficient_llm/warm_order/router_screen.py scripts/loop_screen_study.py scripts/router_screen_study.py \
  --module scripts.moe_dyn_study \
  --handoff $out/router_screen_20261123_a01/native-verified.json \
  --handoff-sha256 bc052501d2ea3e8ce2bdb7ebaa8f067f6a8d405d63a334ee166bb555ac1a5188

/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/moe_dyn_20261123_a01')
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
