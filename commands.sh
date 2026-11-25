#1 +240+a
#th2-ohmv-dot-t-vs-loops-t01-20261125
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
date -u
out=/mnt/local/_outputs/deep-llms_th2
if [ -e "$out/dot_t_loops_20261125_t01" ]; then echo T01_ALREADY_LAUNCHED; cat $out/dot_t_loops_20261125_t01/status.json || true; exit 0; fi
/usr/bin/python3 - <<'PY'
import json
from pathlib import Path
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_recipe_lm_20261125_a07')
t = json.loads((base/'terminal.json').read_text())
print('A07_TERMINAL', json.dumps(t), flush=True)
assert t['native_verified'] is True, 'native not verified'
PY
receipt=$(sha256sum $out/dot_recipe_lm_20261125_a07/native-verified.json | cut -d' ' -f1)
echo "NATIVE_RECEIPT $receipt"
pkg="$PWD/resources/dot_t01_20261125.tar.xz"
echo "f624d75ac16aa1f421f45825b1cba380fa6958ea410025282dbc6d7b37b2d6c0  $pkg" | sha256sum -c -
extra=$(mktemp -d $out/dot_t01_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
export PYTHONPATH=$out/dot_recipe_lm_20261125_a07/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/dot_t_loops_20261125_t01 \
  --old $out/dot_recipe_lm_20261125_a07/execution --prior-sha256 0ef07e69ae4b1cbc5a6b36f5c6a6836c16195963cad65c5208530f8c9c52fe81 \
  --added efficient_llm/warm_order/loop_baselines.py efficient_llm/warm_order/loop_screen.py scripts/loop_screen_study.py scripts/dot_sandwich_study.py \
  --replaced efficient_llm/warm_order/loop_screen.py scripts/loop_screen_study.py scripts/dot_sandwich_study.py \
  --module scripts.dot_sandwich_study \
  --handoff $out/dot_recipe_lm_20261125_a07/native-verified.json \
  --handoff-sha256 "$receipt"
/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_t_loops_20261125_t01')
jobs = [(0, 'swst_fixed_d3'), (1, 'swst_fixed_d3'), (0, 'loop_cycle_d3'), (1, 'loop_cycle_d3'),
        (0, 'loop_prc_d3'), (1, 'loop_prc_d3'), (0, 'swst_fixed_d3r'), (0, 'loop_cycle_d3r')]
paths = {f's{s}/{a}': base/f's{s}'/a/'progress.json' for s, a in jobs}
for _ in range(36):
    if (base/'failure.json').is_file():
        break
    if len([p for p in paths.values() if p.is_file()]) == len(paths):
        break
    time.sleep(5)
print('T01_FAILURE', (base/'failure.json').read_text()[-3000:] if (base/'failure.json').is_file() else 'none', flush=True)
print('T01_PROGRESS', json.dumps({a: json.loads(p.read_text()) for a, p in paths.items() if p.is_file()}), flush=True)
print('T01_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
