#1 +240+a
#th2-ohmv-dot-f18b-20261126
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
date -u
out=/mnt/local/_outputs/deep-llms_th2
if [ -e "$out/dot_f18b_20261126" ]; then echo F18B_ALREADY_LAUNCHED; cat $out/dot_f18b_20261126/status.json || true; exit 0; fi
/usr/bin/python3 - <<'PY'
import json
from pathlib import Path
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_f18_20261126_run')
t = json.loads((base/'terminal.json').read_text())
print('F18_FAILED_TERMINAL', json.dumps(t), flush=True)
assert t['native_verified'] is True, 'native not verified'
PY
receipt=$(sha256sum $out/dot_f18_20261126_run/native-verified.json | cut -d' ' -f1)
echo "NATIVE_RECEIPT $receipt"
prior=21f77857d2bab3219f6a0dd650562c8d6b2d9d753417dafe442544bc1a16b3e0
pkg="$PWD/resources/dot_f18b_20261126.tar.xz"
echo "c870915519d25726ed126c5579c990c2d826753b60119eda8175b86bc460d664  $pkg" | sha256sum -c -
extra=$(mktemp -d $out/dot_f18b_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
export PYTHONPATH=$out/dot_f18_20261126_run/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/dot_f18b_20261126 \
  --old $out/dot_f18_20261126_run/execution --prior-sha256 "$prior" \
  --added efficient_llm/warm_order/dot_stop_sandwich.py \
  --replaced efficient_llm/warm_order/dot_stop_sandwich.py \
  --module scripts.dot_sandwich_study \
  --handoff $out/dot_f18_20261126_run/native-verified.json \
  --handoff-sha256 "$receipt"
/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_f18b_20261126')
jobs = [(0, 'loop_attnres_t512_fa'), (1, 'loop_attnres_t512_fa'), (0, 'loop_attnres_p4_fa'), (1, 'loop_attnres_p4_fa'),
        (0, 'swst_d7_tmlp_ff512_rand_lr14_fa'), (1, 'swst_d7_tmlp_ff512_rand_lr14_fa'),
        (0, 'swst_d7_tmlp_ff512_rand_noid_lr14_fa'), (1, 'swst_d7_tmlp_ff512_rand_noid_lr14_fa')]
paths = {f's{s}/{a}': base/f's{s}'/a/'progress.json' for s, a in jobs}
for _ in range(36):
    if (base/'failure.json').is_file():
        break
    if len([p for p in paths.values() if p.is_file()]) == len(paths):
        break
    time.sleep(5)
print('F18B_FAILURE', (base/'failure.json').read_text()[-3000:] if (base/'failure.json').is_file() else 'none', flush=True)
print('F18B_PROGRESS', json.dumps({a: json.loads(p.read_text()) for a, p in paths.items() if p.is_file()}), flush=True)
print('F18B_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
