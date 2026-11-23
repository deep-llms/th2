#1 +240+a
#th2-ohmv-translator-screen-20261123-a01
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
pkg="$PWD/resources/translator_20261123.tar.xz"
echo "04fdc08c9e7fc9d8bd4a7822e28d3c0a2ac6fc42530fbfdf9230448cdb1b134f  $pkg" | sha256sum -c -
extra=$(mktemp -d /mnt/local/_outputs/deep-llms_th2/translator_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
out=/mnt/local/_outputs/deep-llms_th2
export PYTHONPATH=$out/sandwich_20261123_a01/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/translator_screen_20261123_a01 \
  --old $out/sandwich_20261123_a01/execution --prior-sha256 17159a1d3776b6e5346d06e9bc6c0e067694e75482dffd84860adbf70a6068b7 \
  --added efficient_llm/warm_order/translator_screen.py scripts/translator_screen_study.py \
  --module scripts.translator_screen_study \
  --handoff $out/sandwich_20261123_a01/native-verified.json \
  --handoff-sha256 e8fe18d528c8ac0bae8e685df502aa833d8fc4d88444c1cd6fae37530e1a1744

/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/translator_screen_20261123_a01')
arms = [f'{t}_{o}_{m}' for t in ('tied', 'slot') for o in ('fixed', 'mix') for m in ('residual', 'attnres')]
for _ in range(40):
    if (base/'failure.json').is_file():
        print('TRANSLATOR_FAILURE', (base/'failure.json').read_text()[-2000:], flush=True)
        break
    prog = {a: json.loads(p.read_text()) for a in arms if (p := base/a/'progress.json').is_file()}
    if len(prog) == len(arms):
        print('TRANSLATOR_PROGRESS', json.dumps(prog), flush=True)
        break
    time.sleep(5)
print('TRANSLATOR_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
