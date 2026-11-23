#1 +240+a
#th2-ohmv-domain-screen-20261123-a01
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
pkg="$PWD/resources/domain_20261123.tar.xz"
echo "29427f561b0ff08c0d526c14587068f4871a8e9c300f11d796a3e0d0e2d12218  $pkg" | sha256sum -c -
extra=$(mktemp -d /mnt/local/_outputs/deep-llms_th2/domain_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
out=/mnt/local/_outputs/deep-llms_th2
ls -la /mnt/local/_data/deep-llms_th2/a2_dot_20261118/corpora/fineweb_edu/packed/fineweb_edu/
export PYTHONPATH=$out/translator_screen_s1_20261123_a01/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/domain_screen_20261123_a01 \
  --old $out/translator_screen_s1_20261123_a01/execution --prior-sha256 1411ebd47b3c3a9cee081533e1948db49b1c125f89e7a582f41b8acf45043f97 \
  --added efficient_llm/warm_order/domain_screen.py scripts/domain_screen_study.py \
  --module scripts.domain_screen_study \
  --handoff $out/translator_screen_s1_20261123_a01/native-verified.json \
  --handoff-sha256 c542432217a178b51633dc8c673b8fede32e1d4960490385e1bb93e386fb333b

/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/domain_screen_20261123_a01')
for _ in range(44):
    if (base/'failure.json').is_file():
        print('DOMAIN_FAILURE', (base/'failure.json').read_text()[-2000:], flush=True)
        break
    prog = {a: json.loads(p.read_text()) for a in ('single', 'oracle', 'random', 'both') if (p := base/a/'progress.json').is_file()}
    if len(prog) == 4:
        print('DOMAIN_PROGRESS', json.dumps(prog), flush=True)
        break
    time.sleep(5)
for n in ('status.json', 'prepared.json'):
    p = base/n
    print('DOMAIN_'+n, p.read_text()[:3000] if p.is_file() else 'absent', flush=True)
PROGRESS
