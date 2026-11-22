#1 +180+a
#th2-ohmv-anchor-static-20261123-a01
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
pkg="$PWD/resources/anchor_static_20261123.tar.xz"
echo "e1340149e6a52cdb0283032ed36a6e9d6f3ebf2383254bd0ef56f2c1ef209e18  $pkg" | sha256sum -c -
extra=$(mktemp -d /mnt/local/_outputs/deep-llms_th2/anchor_static_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
export PYTHONPATH=/mnt/local/_outputs/deep-llms_th2/anchor_loop_20261123_a01/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_anchor_long.py" "$extra" --static

/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json,time
base=Path('/mnt/local/_outputs/deep-llms_th2/anchor_static_20261123_a01')
for _ in range(28):
    if (base/'failure.json').is_file():
        raise RuntimeError((base/'failure.json').read_text())
    paths=[base/f's{s}/static/progress.json' for s in (0,1)]
    if all(p.is_file() for p in paths):
        print('STATIC_FIT_PROGRESS', [json.loads(p.read_text()) for p in paths], flush=True)
        break
    time.sleep(5)
PROGRESS
