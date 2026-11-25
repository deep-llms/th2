#1 +240+a
#th2-ohmv-read-f13-then-dot-t-ablation-f14-20261126
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
date -u
out=/mnt/local/_outputs/deep-llms_th2
if [ -e "$out/dot_t_ablation_20261126_f14" ]; then echo F14_ALREADY_LAUNCHED; cat $out/dot_t_ablation_20261126_f14/status.json || true; exit 0; fi
/usr/bin/python3 - <<'PY'
import json, time
from pathlib import Path
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_vs_loops_2x_20261126_f13')
for i in range(360):  # no signal: wait up to 30 min for f13 to end on its own
    if (base/'terminal.json').exists() or (base/'failure.json').exists():
        break
    time.sleep(5)
def j(p):
    return json.loads(p.read_text()) if p.is_file() else None
jobs = [(0, 'loop_cycle_a8_fa_2x'), (1, 'loop_cycle_a8_fa_2x'), (0, 'loop_cycle_a8_lr14_fa_2x'), (1, 'loop_cycle_a8_lr14_fa_2x'),
        (0, 'loop_prc_a8_fa_2x'), (1, 'loop_prc_a8_fa_2x'), (0, 'dense4_fa_2x'), (1, 'dense4_fa_2x')]
r = {}
for s, a in jobs:
    ds = j(base/'downstream'/f's{s}-{a}'/'summary.json')
    r[f's{s}/{a}'] = dict(nll=(j(base/f's{s}'/a/'summary.json') or {}).get('nll'),
                          downstream={t: round(v['accuracy'], 4) for t, v in ds['tasks'].items()} if ds else None)
print('F13_RESULTS', json.dumps(r), flush=True)
print('F13_FAILURE_PRE', (base/'failure.json').read_text()[-1500:] if (base/'failure.json').exists() else 'none', flush=True)
t = json.loads((base/'terminal.json').read_text())
print('F13_TERMINAL', json.dumps(t), flush=True)
assert t['native_verified'] is True, 'native not verified'
PY
receipt=$(sha256sum $out/dot_vs_loops_2x_20261126_f13/native-verified.json | cut -d' ' -f1)
echo "NATIVE_RECEIPT $receipt"
prior=69006b6064052ca33f473fa1fca4c9b0d6bd76386a08e4a8029626ace33b1739
pkg="$PWD/resources/dot_f14_20261126.tar.xz"
echo "6993ad6c99788fa914bad246a8474f2bea55785cac17e1747e76791a52c6c1b2  $pkg" | sha256sum -c -
extra=$(mktemp -d $out/dot_f14_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
export PYTHONPATH=$out/dot_vs_loops_2x_20261126_f13/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/dot_t_ablation_20261126_f14 \
  --old $out/dot_vs_loops_2x_20261126_f13/execution --prior-sha256 "$prior" \
  --added efficient_llm/warm_order/dot_stop_sandwich.py efficient_llm/warm_order/loop_baselines.py efficient_llm/warm_order/loop_screen.py scripts/dot_sandwich_study.py \
  --replaced efficient_llm/warm_order/dot_stop_sandwich.py efficient_llm/warm_order/loop_baselines.py efficient_llm/warm_order/loop_screen.py scripts/dot_sandwich_study.py \
  --module scripts.dot_sandwich_study \
  --handoff $out/dot_vs_loops_2x_20261126_f13/native-verified.json \
  --handoff-sha256 "$receipt"
/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_t_ablation_20261126_f14')
jobs = [(0, 'swst_d7_tmlp_ff512_noid_lr14_fa'), (1, 'swst_d7_tmlp_ff512_noid_lr14_fa'), (0, 'loop_cycle_a7_inj_lr14_fa'),
        (1, 'loop_cycle_a7_inj_lr14_fa'), (0, 'loop_cycle_a8_step_lr14_fa'), (1, 'loop_cycle_a8_step_lr14_fa'),
        (0, 'loop_cycle_a8_lr14_fa'), (0, 'swst_d7_tmlp_ff512_unshared_lr14_fa')]
paths = {f's{s}/{a}': base/f's{s}'/a/'progress.json' for s, a in jobs}
for _ in range(36):
    if (base/'failure.json').is_file():
        break
    if len([p for p in paths.values() if p.is_file()]) == len(paths):
        break
    time.sleep(5)
print('F14_FAILURE', (base/'failure.json').read_text()[-3000:] if (base/'failure.json').is_file() else 'none', flush=True)
print('F14_PROGRESS', json.dumps({a: json.loads(p.read_text()) for a, p in paths.items() if p.is_file()}), flush=True)
print('F14_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
