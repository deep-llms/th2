#1 +240+a
#th2-ohmv-wait-s03-then-dot-synth-s04-20261124
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
date -u
out=/mnt/local/_outputs/deep-llms_th2
if [ -e "$out/dot_synth_20261124_s04" ]; then echo S04_ALREADY_LAUNCHED; cat $out/dot_synth_20261124_s04/status.json || true; exit 0; fi
# 1) No signal: wait for s03 to end on its own; print its compact results.
/usr/bin/python3 - <<'PY'
import json, time
from pathlib import Path
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_synth_20261124_s03')
for i in range(1320):  # up to 110 min
    if (base/'terminal.json').exists() or (base/'failure.json').exists():
        break
    time.sleep(5)
print('S03_FAILURE', (base/'failure.json').read_text()[-1500:] if (base/'failure.json').exists() else 'none', flush=True)
t = json.loads((base/'terminal.json').read_text())
print('S03_TERMINAL', json.dumps(t), flush=True)
m = base/'export'/'manifest.json'
print('S03_EXPORT', {k: v for k, v in json.loads(m.read_text()).items() if k != 'members'} if m.exists() else 'absent', flush=True)
compact = {}
for s in sorted(base.glob('s0/*/*/summary.json')):
    v = json.loads(s.read_text())
    compact[f"{v['arm']}/{v['task']}"] = dict(apps=v['applications'], secs=round(v['train_seconds']),
        res={st: {k: [round(r['exact'], 3), round(r['token'], 3), round(r.get('answer_depth', -1), 2)] for k, r in res.items()}
             for st, res in v['results'].items()})
print('SYNTH_RESULTS', json.dumps(compact, separators=(',', ':')), flush=True)
assert t['native_verified'] is True, 'native not verified'
PY
receipt=$(sha256sum $out/dot_synth_20261124_s03/native-verified.json | cut -d' ' -f1)
echo "NATIVE_RECEIPT $receipt"
# 2) Launch s04 (opseq replication, seeds 1-2); old source = s03 execution, pinned hash.
pkg="$PWD/resources/dot_synth_s04_20261124.tar.xz"
echo "e74fbad38116b11adf7f11ecff5be2d3b28ff771a3e323c45f8c65be34d9750d  $pkg" | sha256sum -c -
extra=$(mktemp -d $out/dot_synth_s04_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
export PYTHONPATH=$out/dot_synth_20261124_s03/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/dot_synth_20261124_s04 \
  --old $out/dot_synth_20261124_s03/execution --prior-sha256 611f93afd56897c18ff4cc7c4905379e0efaac36e8bd1668ea0fd898bcf6fec3 \
  --added scripts/dot_synth_study.py \
  --replaced scripts/dot_synth_study.py \
  --module scripts.dot_synth_study \
  --handoff $out/dot_synth_20261124_s03/native-verified.json \
  --handoff-sha256 "$receipt"
/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_synth_20261124_s04')
for _ in range(24):
    if (base/'failure.json').is_file() or (base/'status.json').is_file() and 'train' in (base/'status.json').read_text():
        break
    time.sleep(5)
print('S04_FAILURE', (base/'failure.json').read_text()[-3000:] if (base/'failure.json').is_file() else 'none', flush=True)
print('S04_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
