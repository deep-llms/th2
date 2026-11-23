#1 +240+a
#th2-ohmv-sandwich-20261123-a01
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
pkg="$PWD/resources/sandwich_20261123.tar.xz"
echo "9248b07d9dca11ca9557612fee303052eb57c4505b4ecfd46af9a1389caae05c  $pkg" | sha256sum -c -
extra=$(mktemp -d /mnt/local/_outputs/deep-llms_th2/sandwich_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
out=/mnt/local/_outputs/deep-llms_th2
export PYTHONPATH=$out/loop_stop2_20261123_a01/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/sandwich_20261123_a01 \
  --old $out/loop_stop2_20261123_a01/execution --prior-sha256 2dbca58304be81cfeab67ec3da162de5e6b2b058e1450fe93e6af0e0678e5c42 \
  --added efficient_llm/warm_order/sandwich.py scripts/sandwich_study.py \
  --module scripts.sandwich_study \
  --handoff $out/loop_screen4_downstream_20261123_a01/native-verified.json \
  --handoff-sha256 39f94a2b8e9c8978e22dc94cbbe3e6e791d69ca2ec20aaa931a4a6a6065d06c4

/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/sandwich_20261123_a01')
for _ in range(40):
    if (base/'failure.json').is_file():
        print('SANDWICH_FAILURE', (base/'failure.json').read_text()[-2000:], flush=True)
        break
    prog = {n: json.loads(p.read_text()) for n in ('sw_warm', 'not_warm') if (p := base/n/'progress.json').is_file()}
    if len(prog) == 2:
        print('SANDWICH_PROGRESS', json.dumps(prog), flush=True)
        break
    time.sleep(5)
print('SANDWICH_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
