#1 +240+a
#th2-ohmv-moe-variants-20261123-a01
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
pkg="$PWD/resources/moe_variants_20261123.tar.xz"
echo "bbbba3be97d9726854ae36083e12d57a4bb519a1203d9127ac517b49e395fb31  $pkg" | sha256sum -c -
extra=$(mktemp -d /mnt/local/_outputs/deep-llms_th2/moe_variants_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
out=/mnt/local/_outputs/deep-llms_th2
export PYTHONPATH=$out/moe_downstream_20261123_a01/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/moe_variants_20261123_a01 \
  --old $out/moe_downstream_20261123_a01/execution --prior-sha256 5b2408b14a317a49da38e9aea5ea60802ec10923bebdb747a7a2d7f0afc34bf8 \
  --added efficient_llm/warm_order/loop_screen.py scripts/loop_screen_study.py scripts/moe_variants_study.py \
  --replaced efficient_llm/warm_order/loop_screen.py scripts/loop_screen_study.py \
  --module scripts.moe_variants_study \
  --handoff $out/moe_downstream_20261123_a01/native-verified.json \
  --handoff-sha256 5b5eace1df70bdd5eb2972fd991be406acd0fe0aea8fb33f3016e856e9713e5f

/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/moe_variants_20261123_a01')
jobs = [(0, 'loop_attnres_moe_pr'), (1, 'loop_attnres_moe_pr'), (0, 'loop_attnres_moe'), (1, 'loop_attnres_moe'),
        (0, 'attnres_moe16'), (1, 'attnres_moe16'), (0, 'attnres_moe_shared'), (1, 'attnres_moe_shared')]
paths = {f's{s}/{a}': base/f's{s}'/a/'progress.json' for s, a in jobs}
for _ in range(44):
    if (base/'failure.json').is_file():
        print('MOE_VAR_FAILURE', (base/'failure.json').read_text()[-2000:], flush=True)
        break
    prog = {a: json.loads(p.read_text()) for a, p in paths.items() if p.is_file()}
    if len(prog) == len(paths):
        print('MOE_VAR_PROGRESS', json.dumps(prog), flush=True)
        break
    time.sleep(5)
print('MOE_VAR_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
