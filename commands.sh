#1 +240+a
#th2-ohmv-dot-synth-s06-20261124
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
date -u
out=/mnt/local/_outputs/deep-llms_th2
if [ -e "$out/dot_synth_20261124_s06" ]; then echo S06_ALREADY_LAUNCHED; cat $out/dot_synth_20261124_s06/status.json || true; exit 0; fi
# 1) s05 has ended (07:26Z); confirm its terminal and print its compact results.
/usr/bin/python3 - <<'PY'
import json, time
from pathlib import Path
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_synth_20261124_s05')
for i in range(60):  # s05 already ended; short guard
    if (base/'terminal.json').exists() or (base/'failure.json').exists():
        break
    time.sleep(5)
print('S05_FAILURE', (base/'failure.json').read_text()[-1500:] if (base/'failure.json').exists() else 'none', flush=True)
t = json.loads((base/'terminal.json').read_text())
print('S05_TERMINAL', json.dumps(t), flush=True)
m = base/'export'/'manifest.json'
print('S05_EXPORT', {k: v for k, v in json.loads(m.read_text()).items() if k != 'members'} if m.exists() else 'absent', flush=True)
compact = {}
for s in sorted(base.glob('s*/*/*/summary.json')):
    v = json.loads(s.read_text())
    compact[f"s{v['seed']}/{v['arm']}/{v['task']}"] = dict(apps=v['applications'], secs=round(v['train_seconds']),
        res={st: {k: [round(r['exact'], 3), round(r['token'], 3), round(r.get('answer_depth', -1), 2)] for k, r in res.items()}
             for st, res in v['results'].items()},
        routes={k: {q: r['routes'].get(q) for q in ('distinct', 'top', 'modal_given_key')} for k, r in v['results']['native'].items()
                if 'routes' in r})
print('SYNTH_RESULTS', json.dumps(compact, separators=(',', ':')), flush=True)
assert t['native_verified'] is True, 'native not verified'
PY
receipt=$(sha256sum $out/dot_synth_20261124_s05/native-verified.json | cut -d' ' -f1)
echo "NATIVE_RECEIPT $receipt"
# 2) Launch s06 (chain: fixed-order prior routing, third seed, confidence halting); old source = s05 execution.
pkg="$PWD/resources/dot_synth_s06_20261124.tar.xz"
echo "76d84e3f10c2a7c87d96caecd3e17740289f971003d1364963b67593e964b6a4  $pkg" | sha256sum -c -
extra=$(mktemp -d $out/dot_synth_s06_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
export PYTHONPATH=$out/dot_synth_20261124_s05/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/dot_synth_20261124_s06 \
  --old $out/dot_synth_20261124_s05/execution --prior-sha256 5e3f2747d0857d91c9fe9e2df69cf792a53a887fe406735164f1a7d2974c611b \
  --added efficient_llm/warm_order/dot_stop_sandwich.py efficient_llm/warm_order/loop_screen.py scripts/dot_synth_study.py \
  --replaced efficient_llm/warm_order/dot_stop_sandwich.py efficient_llm/warm_order/loop_screen.py scripts/dot_synth_study.py \
  --module scripts.dot_synth_study \
  --handoff $out/dot_synth_20261124_s05/native-verified.json \
  --handoff-sha256 "$receipt"
/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_synth_20261124_s06')
for _ in range(24):
    if (base/'failure.json').is_file() or (base/'status.json').is_file() and 'train' in (base/'status.json').read_text():
        break
    time.sleep(5)
print('S06_FAILURE', (base/'failure.json').read_text()[-3000:] if (base/'failure.json').is_file() else 'none', flush=True)
print('S06_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
