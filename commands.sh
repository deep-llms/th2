#1 +240+a
#th2-ohmv-dot-headroom2-20261124-a01
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
out=/mnt/local/_outputs/deep-llms_th2
/usr/bin/python3 -c "import json; t=json.load(open('$out/dot_headroom_20261124_a01/terminal.json')); assert t['native_verified'] is True, t; print('PRIOR_TERMINAL', t)"
receipt=$(sha256sum $out/dot_headroom_20261124_a01/native-verified.json | cut -d' ' -f1)
echo "NATIVE_RECEIPT $receipt"
pkg="$PWD/resources/dot_headroom2_20261124.tar.xz"
echo "399b782214c017e8571e5438c4c4862a906878ef6a4891547ad1bac4e6fe5de7  $pkg" | sha256sum -c -
extra=$(mktemp -d $out/dot_headroom2_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
export PYTHONPATH=$out/dot_headroom_20261124_a01/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/dot_headroom2_20261124_a01 \
  --old $out/dot_headroom_20261124_a01/execution --prior-sha256 cfaf46bb7628876292550f3d4efaa3767d0affaca66712faed776be587924bae \
  --added scripts/dot_headroom2_study.py \
  --module scripts.dot_headroom2_study \
  --handoff $out/dot_headroom_20261124_a01/native-verified.json \
  --handoff-sha256 "$receipt"
/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_headroom2_20261124_a01')
for _ in range(180):  # evaluation only: wait for the terminal (up to 15 min)
    if (base/'terminal.json').is_file() or (base/'failure.json').is_file():
        break
    time.sleep(5)
for n in ('failure.json', 'terminal.json'):
    print(n, (base/n).read_text()[-2500:] if (base/n).is_file() else 'absent', flush=True)
if (base/'failure.json').is_file():
    for p in sorted(base.glob('eval-*.log')):
        t = p.read_text()
        if 'Error' in t[-3000:]:
            print('LOG', p.name, t[-2000:], flush=True)
for p in sorted(base.glob('*/summary.json')):
    print('HEADROOM2', p.read_text(), flush=True)
m = base/'export'/'manifest.json'
print('EXPORT', {k: v for k, v in json.loads(m.read_text()).items() if k != 'members'} if m.exists() else 'absent', flush=True)
PROGRESS
