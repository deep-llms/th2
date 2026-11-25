#1 +240+a
#th2-ohmv-dot-f15-20261126
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
date -u
out=/mnt/local/_outputs/deep-llms_th2
if [ -e "$out/dot_f15_20261126" ]; then echo F15_ALREADY_LAUNCHED; cat $out/dot_f15_20261126/status.json || true; exit 0; fi
/usr/bin/python3 - <<'PY'
import json
from pathlib import Path
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_t_ablation_20261126_f14')
t = json.loads((base/'terminal.json').read_text())
print('F14_TERMINAL', json.dumps(t), flush=True)
assert t['native_verified'] is True, 'native not verified'
PY
receipt=$(sha256sum $out/dot_t_ablation_20261126_f14/native-verified.json | cut -d' ' -f1)
echo "NATIVE_RECEIPT $receipt"
[ "$receipt" = 71733a0dc277ad4594dda04eb22631d71aaefa296dd2dd607e354bf3e297c4a1 ]
prior=66d9751de61d0596bce38c47682e31c2f52e158dab4a8c5779115472a368212f
pkg="$PWD/resources/dot_f15_20261126.tar.xz"
echo "785932fc2ecd4c607f6a3e450698d072b281003cf719b2cd832724ddd23f217f  $pkg" | sha256sum -c -
extra=$(mktemp -d $out/dot_f15_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
export PYTHONPATH=$out/dot_t_ablation_20261126_f14/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/dot_f15_20261126 \
  --old $out/dot_t_ablation_20261126_f14/execution --prior-sha256 "$prior" \
  --added efficient_llm/warm_order/loop_baselines.py efficient_llm/warm_order/loop_screen.py scripts/dot_sandwich_study.py \
  --replaced efficient_llm/warm_order/loop_baselines.py efficient_llm/warm_order/loop_screen.py scripts/dot_sandwich_study.py \
  --module scripts.dot_sandwich_study \
  --handoff $out/dot_t_ablation_20261126_f14/native-verified.json \
  --handoff-sha256 "$receipt"
/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_f15_20261126')
jobs = [(0, 'loop_cycle_a8_injadd_fa'), (1, 'loop_cycle_a8_injadd_fa'), (1, 'swst_d7_tmlp_ff512_unshared_lr14_fa'),
        (0, 'swst_d7_tmlp_ff512_unshared_lr14_fa_2x'), (1, 'swst_d7_tmlp_ff512_unshared_lr14_fa_2x'),
        (0, 'swst_m8d8_tmlp_ff512_fa'), (1, 'swst_m8d8_tmlp_ff512_fa'), (0, 'swst_m8d8_tmlp_ff512_lr14_fa')]
paths = {f's{s}/{a}': base/f's{s}'/a/'progress.json' for s, a in jobs}
for _ in range(36):
    if (base/'failure.json').is_file():
        break
    if len([p for p in paths.values() if p.is_file()]) == len(paths):
        break
    time.sleep(5)
print('F15_FAILURE', (base/'failure.json').read_text()[-3000:] if (base/'failure.json').is_file() else 'none', flush=True)
print('F15_PROGRESS', json.dumps({a: json.loads(p.read_text()) for a, p in paths.items() if p.is_file()}), flush=True)
print('F15_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
