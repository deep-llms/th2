#1 +240+a
#th2-ohmv-wait-f09-then-dot-fast-attn-f10-20261125
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
date -u
out=/mnt/local/_outputs/deep-llms_th2
if [ -e "$out/dot_fast_attn_20261125_f10" ]; then echo F10_ALREADY_LAUNCHED; cat $out/dot_fast_attn_20261125_f10/status.json || true; exit 0; fi
/usr/bin/python3 - <<'PY'
import json, time
from pathlib import Path
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_narrow_tmlp_20261125_f09')
for i in range(720):  # no signal: wait up to 60 min for f09 to end on its own
    if (base/'terminal.json').exists() or (base/'failure.json').exists():
        break
    time.sleep(5)
def j(p):
    return json.loads(p.read_text()) if p.is_file() else None
jobs = [(0, 'swst_d6_tmlp_ff1024'), (1, 'swst_d6_tmlp_ff1024'), (0, 'swst_d7_tmlp_ff512'), (1, 'swst_d7_tmlp_ff512'),
        (0, 'swst_d6_tmlp_ff512'), (1, 'swst_d6_tmlp_ff512'), (0, 'swst_d8_tmlp_ff512'), (1, 'swst_d8_tmlp_ff512')]
r = {}
for s, a in jobs:
    ds = j(base/'downstream'/f's{s}-{a}'/'summary.json')
    r[f's{s}/{a}'] = dict(nll=(j(base/f's{s}'/a/'summary.json') or {}).get('nll'),
                          downstream={t: round(v['accuracy'], 4) for t, v in ds['tasks'].items()} if ds else None)
print('F09_RESULTS', json.dumps(r), flush=True)
print('F09_FAILURE_PRE', (base/'failure.json').read_text()[-1500:] if (base/'failure.json').exists() else 'none', flush=True)
t = json.loads((base/'terminal.json').read_text())
print('F09_TERMINAL', json.dumps(t), flush=True)
assert t['native_verified'] is True, 'native not verified'
PY
receipt=$(sha256sum $out/dot_narrow_tmlp_20261125_f09/native-verified.json | cut -d' ' -f1)
echo "NATIVE_RECEIPT $receipt"
prior=2ff04abc74deba94cea64d2adf4824e65294feb21d74939ca1e64dd950a4ff92
pkg="$PWD/resources/dot_f10_20261125.tar.xz"
echo "ed8db89413f186c10288fdc8981a18473867ce3094873574fb5f4bc8df07a374  $pkg" | sha256sum -c -
extra=$(mktemp -d $out/dot_f10_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
export PYTHONPATH=$out/dot_narrow_tmlp_20261125_f09/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/dot_fast_attn_20261125_f10 \
  --old $out/dot_narrow_tmlp_20261125_f09/execution --prior-sha256 "$prior" \
  --added efficient_llm/dot_pool/model.py efficient_llm/warm_order/adaptation_runtime.py efficient_llm/warm_order/shared_heads.py efficient_llm/warm_order/dot_stop_sandwich.py efficient_llm/warm_order/loop_baselines.py efficient_llm/warm_order/loop_screen.py scripts/loop_screen_study.py scripts/dot_sandwich_study.py \
  --replaced efficient_llm/dot_pool/model.py efficient_llm/warm_order/adaptation_runtime.py efficient_llm/warm_order/shared_heads.py efficient_llm/warm_order/dot_stop_sandwich.py efficient_llm/warm_order/loop_baselines.py efficient_llm/warm_order/loop_screen.py scripts/loop_screen_study.py scripts/dot_sandwich_study.py \
  --module scripts.dot_sandwich_study \
  --handoff $out/dot_narrow_tmlp_20261125_f09/native-verified.json \
  --handoff-sha256 "$receipt"
/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_fast_attn_20261125_f10')
jobs = [(0, 'swst_d6_tmlp_lr14_fa'), (1, 'swst_d6_tmlp_lr14_fa'), (0, 'loop_cycle_a11_lr14_fa'), (1, 'loop_cycle_a11_lr14_fa'),
        (0, 'dense8_lr14'), (1, 'dense8_lr14'), (0, 'dense8_lr14_fa'), (1, 'dense8_lr14_fa')]
paths = {f's{s}/{a}': base/f's{s}'/a/'progress.json' for s, a in jobs}
for _ in range(36):
    if (base/'failure.json').is_file():
        break
    if len([p for p in paths.values() if p.is_file()]) == len(paths):
        break
    time.sleep(5)
print('F10_FAILURE', (base/'failure.json').read_text()[-3000:] if (base/'failure.json').is_file() else 'none', flush=True)
print('F10_PROGRESS', json.dumps({a: json.loads(p.read_text()) for a, p in paths.items() if p.is_file()}), flush=True)
print('F10_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
