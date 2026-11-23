#1 +240+a
#th2-ohmv-translator-screen-s1-20261123-a01
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
pkg="$PWD/resources/translator_s1_20261123.tar.xz"
echo "36e1199f551892820a70fbf69cbe18f3a0770cf2be4545936b28cf9dd2fd5411  $pkg" | sha256sum -c -
extra=$(mktemp -d /mnt/local/_outputs/deep-llms_th2/translator_s1_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
out=/mnt/local/_outputs/deep-llms_th2
export PYTHONPATH=$out/translator_screen_20261123_a01/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/translator_screen_s1_20261123_a01 \
  --old $out/translator_screen_20261123_a01/execution --prior-sha256 4aa99afbfd5e407c21ecf64a5292415dd46235c9ac95a7748ec7f2e55ccf441f \
  --added scripts/translator_screen_study.py --replaced scripts/translator_screen_study.py \
  --module scripts.translator_screen_study \
  --handoff $out/translator_screen_20261123_a01/native-verified.json \
  --handoff-sha256 b156ecaec7208b270575678ab1574f3cd1be618bb063adbdc600bcf423d84dc6

/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/translator_screen_s1_20261123_a01')
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
