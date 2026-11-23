#1 +240+a
#th2-ohmv-router-screen-20261123-a01
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
pkg="$PWD/resources/router_20261123.tar.xz"
echo "067a44a859be2e4b6a4f7938d170a89f6b1533b219278f8db2971b26ca9feccf  $pkg" | sha256sum -c -
extra=$(mktemp -d /mnt/local/_outputs/deep-llms_th2/router_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
out=/mnt/local/_outputs/deep-llms_th2
export PYTHONPATH=$out/family_downstream_20261123_a01/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/router_screen_20261123_a01 \
  --old $out/family_downstream_20261123_a01/execution --prior-sha256 9cbc7e20027ea2cb5e46269ad4d7f4744e2fa47cac41a92ad6bb3689eed0f1a7 \
  --added efficient_llm/warm_order/router_screen.py scripts/router_screen_study.py \
  --module scripts.router_screen_study \
  --handoff $out/family_downstream_20261123_a01/native-verified.json \
  --handoff-sha256 d98b9da901677f2f8e0f541f646ab0b19587865ac26f35e2567077bcb7222202

/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/router_screen_20261123_a01')
arms = ('single', 'seq_switch2', 'seq_switch3', 'seq_cf2', 'order_cf', 'both_mix', 'tok_moe2', 'tok_moe3')
for _ in range(44):
    if (base/'failure.json').is_file():
        print('ROUTER_FAILURE', (base/'failure.json').read_text()[-2000:], flush=True)
        break
    prog = {a: json.loads((base/a/'progress.json').read_text()) for a in arms if (base/a/'progress.json').is_file()}
    if len(prog) == len(arms):
        print('ROUTER_PROGRESS', json.dumps(prog), flush=True)
        break
    time.sleep(5)
print('ROUTER_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
