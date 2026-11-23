#1 +240+a
#th2-ohmv-dot-headroom-20261124-a01
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
pkg="$PWD/resources/dot_headroom_20261124.tar.xz"
echo "469ed4ed0a863f76d67101a806246bc7190ccb9f4823e907c2671e03a2d506bf  $pkg" | sha256sum -c -
extra=$(mktemp -d /mnt/local/_outputs/deep-llms_th2/dot_headroom_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
out=/mnt/local/_outputs/deep-llms_th2
export PYTHONPATH=$out/dot_sandwich_20261124_a04/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/dot_headroom_20261124_a01 \
  --old $out/dot_sandwich_20261124_a04/execution --prior-sha256 a0982a4de2f190e1bfc762d29c3fea2253cc5c4646678bbe481983c922e96d6c \
  --added efficient_llm/warm_order/dot_sandwich.py efficient_llm/warm_order/loop_screen.py scripts/dot_headroom_study.py \
  --replaced efficient_llm/warm_order/dot_sandwich.py efficient_llm/warm_order/loop_screen.py \
  --module scripts.dot_headroom_study \
  --handoff $out/dot_sandwich_20261124_a05/native-verified.json \
  --handoff-sha256 53754cd43f3de0bc6cbba59d8e8ec67e9b7f6ef187de18b7198215d819fd5677
/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_headroom_20261124_a01')
for _ in range(150):  # evaluation only: wait for the terminal (up to 12.5 min)
    if (base/'terminal.json').is_file() or (base/'failure.json').is_file():
        break
    time.sleep(5)
for n in ('status.json', 'failure.json', 'terminal.json'):
    print(n, (base/n).read_text()[-2000:] if (base/n).is_file() else 'absent', flush=True)
for p in sorted(base.glob('*/summary.json')):
    s = json.loads(p.read_text())
    print('HEADROOM', json.dumps({k: v for k, v in s.items() if k != 'path_means'}), flush=True)
m = base/'export'/'manifest.json'
print('EXPORT', {k: v for k, v in json.loads(m.read_text()).items() if k != 'members'} if m.exists() else 'absent', flush=True)
PROGRESS
