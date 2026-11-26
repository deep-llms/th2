#1 +240+a
#th2-ohmv-dot-f19-20261126
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
date -u
out=/mnt/local/_outputs/deep-llms_th2
if [ -e "$out/dot_f19_20261126" ]; then echo F19_ALREADY_LAUNCHED; cat $out/dot_f19_20261126/status.json || true; exit 0; fi
/usr/bin/python3 - <<'PY'
import json
from pathlib import Path
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_f18b_20261126')
t = json.loads((base/'terminal.json').read_text())
print('F18B_TERMINAL', json.dumps(t), flush=True)
assert t['native_verified'] is True and t['success'] is True, 'f18b did not finish with the native burn verified'
PY
receipt=60957b793a0225f626ee32a6d611dc079bd7d1952a566df9a81890ab32e5639f
echo "$receipt  $out/dot_f18b_20261126/native-verified.json" | sha256sum -c -
prior=b6923744bf9508b26a1da934e9f0f5fcfb011f208146d994ead4d4864cde6065
pkg="$PWD/resources/dot_f19_20261126.tar.xz"
echo "7ace75954247c814c91efb747947ae0bc60583d718551dfc5d0aea052dc184d0  $pkg" | sha256sum -c -
extra=$(mktemp -d $out/dot_f19_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
export PYTHONPATH=$out/dot_f18b_20261126/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/dot_f19_20261126 \
  --old $out/dot_f18b_20261126/execution --prior-sha256 "$prior" \
  --added efficient_llm/warm_order/full_methods.py efficient_llm/warm_order/loop_screen.py efficient_llm/warm_order/loop_screen_downstream.py scripts/loop_screen_study.py scripts/loop_screen_downstream_study.py scripts/dot_sandwich_study.py \
  --replaced efficient_llm/warm_order/loop_screen.py efficient_llm/warm_order/loop_screen_downstream.py scripts/loop_screen_study.py scripts/loop_screen_downstream_study.py scripts/dot_sandwich_study.py \
  --module scripts.dot_sandwich_study \
  --handoff $out/dot_f18b_20261126/native-verified.json \
  --handoff-sha256 "$receipt"
/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_f19_20261126')
jobs = [(0, 'huginn_r3'), (0, 'huginn_r3_lr05'), (0, 'ouro_t2'), (0, 'ouro_t2_lr05'),
        (0, 'mor_mc3'), (0, 'mor_mc3_lr20'), (0, 'huginn_r32'), (0, 'ouro_t4')]
paths = {f's{s}/{a}': base/f's{s}'/a/'progress.json' for s, a in jobs}
for _ in range(48):
    if (base/'failure.json').is_file():
        break
    if len([p for p in paths.values() if p.is_file()]) == len(paths):
        break
    time.sleep(5)
print('F19_FAILURE', (base/'failure.json').read_text()[-3000:] if (base/'failure.json').is_file() else 'none', flush=True)
print('F19_PROGRESS', json.dumps({a: json.loads(p.read_text()) for a, p in paths.items() if p.is_file()}), flush=True)
print('F19_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
