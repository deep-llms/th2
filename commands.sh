#1 +240+a
#th2-ohmv-read-f08-then-dot-narrow-tmlp-f09-20261125
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
date -u
out=/mnt/local/_outputs/deep-llms_th2
if [ -e "$out/dot_narrow_tmlp_20261125_f09" ]; then echo F09_ALREADY_LAUNCHED; cat $out/dot_narrow_tmlp_20261125_f09/status.json || true; exit 0; fi
/usr/bin/python3 - <<'PY'
import json, time
from pathlib import Path
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_lr_high_20261125_f08')
for i in range(360):  # no signal: wait up to 30 min for f08 to end on its own
    if (base/'terminal.json').exists() or (base/'failure.json').exists():
        break
    time.sleep(5)
def j(p):
    return json.loads(p.read_text()) if p.is_file() else None
jobs = [(0, 'loop_cycle_a11_lr14'), (1, 'loop_cycle_a11_lr14'), (0, 'swst_d6_tmlp_lr14'), (1, 'swst_d6_tmlp_lr14'),
        (0, 'loop_cycle_a11_lr20'), (0, 'swst_d6_tmlp_lr20'), (1, 'loop_cycle_a11_lr07'), (1, 'swst_d6_tmlp_lr07')]
r = {}
for s, a in jobs:
    ds = j(base/'downstream'/f's{s}-{a}'/'summary.json')
    r[f's{s}/{a}'] = dict(nll=(j(base/f's{s}'/a/'summary.json') or {}).get('nll'),
                          downstream={t: round(v['accuracy'], 4) for t, v in ds['tasks'].items()} if ds else None)
print('F08_RESULTS', json.dumps(r), flush=True)
print('F08_FAILURE', (base/'failure.json').read_text()[-1500:] if (base/'failure.json').exists() else 'none', flush=True)
t = json.loads((base/'terminal.json').read_text())
print('F08_TERMINAL', json.dumps(t), flush=True)
assert t['native_verified'] is True, 'native not verified'
PY
receipt=$(sha256sum $out/dot_lr_high_20261125_f08/native-verified.json | cut -d' ' -f1)
echo "NATIVE_RECEIPT $receipt"
prior=ed1de3f27f230d9445d83a8d99e8d617419b476727aebee319b50aa282465f0e
pkg="$PWD/resources/dot_f09_20261125.tar.xz"
echo "6fdb99d7649ac07f268d64c13ec1934578d1e342e360971cff1f1cbb21c654f6  $pkg" | sha256sum -c -
extra=$(mktemp -d $out/dot_f09_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
export PYTHONPATH=$out/dot_lr_high_20261125_f08/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/dot_narrow_tmlp_20261125_f09 \
  --old $out/dot_lr_high_20261125_f08/execution --prior-sha256 "$prior" \
  --added efficient_llm/warm_order/dot_stop_sandwich.py efficient_llm/warm_order/loop_screen.py scripts/dot_sandwich_study.py \
  --replaced efficient_llm/warm_order/dot_stop_sandwich.py efficient_llm/warm_order/loop_screen.py scripts/dot_sandwich_study.py \
  --module scripts.dot_sandwich_study \
  --handoff $out/dot_lr_high_20261125_f08/native-verified.json \
  --handoff-sha256 "$receipt"
/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_narrow_tmlp_20261125_f09')
jobs = [(0, 'swst_d6_tmlp_ff1024'), (1, 'swst_d6_tmlp_ff1024'), (0, 'swst_d7_tmlp_ff512'), (1, 'swst_d7_tmlp_ff512'),
        (0, 'swst_d6_tmlp_ff512'), (1, 'swst_d6_tmlp_ff512'), (0, 'swst_d8_tmlp_ff512'), (1, 'swst_d8_tmlp_ff512')]
paths = {f's{s}/{a}': base/f's{s}'/a/'progress.json' for s, a in jobs}
for _ in range(36):
    if (base/'failure.json').is_file():
        break
    if len([p for p in paths.values() if p.is_file()]) == len(paths):
        break
    time.sleep(5)
print('F09_FAILURE', (base/'failure.json').read_text()[-3000:] if (base/'failure.json').is_file() else 'none', flush=True)
print('F09_PROGRESS', json.dumps({a: json.loads(p.read_text()) for a, p in paths.items() if p.is_file()}), flush=True)
print('F09_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
