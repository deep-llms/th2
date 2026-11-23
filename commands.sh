#1 +240+a
#th2-ohmv-stop-then-dot-sandwich-20261123-a01
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
date -u
# 1) User-authorized stop of the owned MoE 4x run (idempotent; same guards as cc2ae7a9), wait for native restore.
/usr/bin/python3 - <<'PY'
import json, os, signal, subprocess, time
from pathlib import Path
base = Path('/mnt/local/_outputs/deep-llms_th2/moe_scale_20261123_a01')
def cmd(pid): return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode().strip()
def ppid(pid): return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[1])
if (base/'terminal.json').exists() or (base/'failure.json').exists():
    print('ALREADY_ENDED', flush=True)
else:
    status = json.loads((base/'status.json').read_text())
    print('STATUS', json.dumps(status), flush=True)
    assert status['stage'] == 'train' and len(status['running']) == 8, 'unexpected stage'
    pane = subprocess.run(['tmux', 'display-message', '-p', '-t', base.name, '#{pane_pid} #{pane_dead}'],
                          capture_output=True, text=True, check=True).stdout.split()
    sup = int(pane[0]); assert pane[1] == '0', 'pane dead'
    c = cmd(sup); print('SUPERVISOR', sup, c, flush=True)
    assert f'-m scripts.moe_scale_study queue --base {base}' in c, 'supervisor command mismatch'
    assert os.readlink(f'/proc/{sup}/cwd') == str(base/'execution'/'source'), 'supervisor cwd mismatch'
    children = {}
    for p in os.listdir('/proc'):
        if p.isdigit():
            try:
                if ppid(int(p)) == sup:
                    children[int(p)] = cmd(int(p))
            except (FileNotFoundError, ProcessLookupError):
                pass
    train = {p for p, cm in children.items() if f'scripts.moe_scale_study train --base {base}' in cm}
    assert len(train) == 8 and set(children) == train, 'children mismatch'
    q = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout
    assert {int(l) for l in q.split()} == train, 'GPU owners are not exactly the eight train workers'
    os.kill(sup, signal.SIGTERM)
    print('SIGTERM_SENT', sup, flush=True)
for _ in range(120):
    if (base/'terminal.json').exists():
        break
    time.sleep(5)
t = json.loads((base/'terminal.json').read_text())
print('TERMINAL', json.dumps(t), flush=True)
assert t['native_verified'] is True, 'native not verified'
PY
out=/mnt/local/_outputs/deep-llms_th2
receipt=$(sha256sum $out/moe_scale_20261123_a01/native-verified.json | cut -d' ' -f1)
echo "NATIVE_RECEIPT $receipt"
# 2) Launch the DOT sandwich round from that receipt.
pkg="$PWD/resources/dot_sandwich_20261123.tar.xz"
echo "a3192ae191124d74c488f22cfc0d3aec8dd67b37f18d169c5e06637fd89344c4  $pkg" | sha256sum -c -
extra=$(mktemp -d $out/dot_sandwich_source.XXXXXXXX)
tar -xJf "$pkg" -C "$extra"
export PYTHONPATH=$out/moe_scale_20261123_a01/execution/source
cd "$PYTHONPATH"
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 "$extra/scripts/launch_loop_screen.py" "$extra" \
  --base $out/dot_sandwich_20261123_a01 \
  --old $out/moe_scale_20261123_a01/execution --prior-sha256 aba8b75a1e8fdaf33ba5ae5d478a7283a16950caf2750ddbf48d8d0c41b959a5 \
  --added efficient_llm/warm_order/dot_sandwich.py efficient_llm/warm_order/dot_token.py efficient_llm/warm_order/loop_screen.py scripts/loop_screen_study.py scripts/loop_screen_downstream_study.py scripts/dot_sandwich_study.py \
  --replaced efficient_llm/warm_order/loop_screen.py scripts/loop_screen_study.py scripts/loop_screen_downstream_study.py \
  --module scripts.dot_sandwich_study \
  --handoff $out/moe_scale_20261123_a01/native-verified.json \
  --handoff-sha256 "$receipt"
/usr/bin/python3 - <<'PROGRESS'
from pathlib import Path
import json, time
base = Path('/mnt/local/_outputs/deep-llms_th2/dot_sandwich_20261123_a01')
jobs = [(0, 'sw_route'), (1, 'sw_route'), (0, 'sw_fixed'), (1, 'sw_fixed'),
        (0, 'sw_random'), (0, 'sw_route_noid'), (0, 'sw_rl'), (0, 'sw_cf')]
paths = {f's{s}/{a}': base/f's{s}'/a/'progress.json' for s, a in jobs}
for _ in range(60):
    if (base/'failure.json').is_file():
        print('DOT_FAILURE', (base/'failure.json').read_text()[-3000:], flush=True)
        break
    prog = {a: json.loads(p.read_text()) for a, p in paths.items() if p.is_file()}
    if len(prog) == len(paths):
        print('DOT_PROGRESS', json.dumps(prog), flush=True)
        break
    time.sleep(5)
print('DOT_STATUS', (base/'status.json').read_text() if (base/'status.json').is_file() else 'absent', flush=True)
PROGRESS
