#1 +240+a
#th2-ohmv-moe-scale-stop-20261123-a01
set -u
date -u
/usr/bin/python3 - <<'PY'
# User-authorized stop (23 Nov ~22:15 SGT) of the owned MoE 4x run moe_scale_20261123_a01 (GPU_SAFETY.md):
# verify the exact supervisor (pane process of its own tmux session), command, cwd, start time, its eight
# train children and their GPU mapping; only then SIGTERM that supervisor PID. Its own handler stops its
# children, waits for free GPUs and restores + verifies the native burn. No group/pattern kills.
import json, os, signal, subprocess, time, hashlib
from pathlib import Path
base = Path('/mnt/local/_outputs/deep-llms_th2/moe_scale_20261123_a01')
def cmd(pid): return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode().strip()
def ppid(pid): return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[1])
status = json.loads((base/'status.json').read_text())
print('STATUS', json.dumps(status), flush=True)
assert status['stage'] == 'train' and len(status['running']) == 8, 'unexpected stage'
assert not (base/'terminal.json').exists() and not (base/'failure.json').exists(), 'run already ended'
pane = subprocess.run(['tmux', 'display-message', '-p', '-t', base.name, '#{pane_pid} #{pane_dead} #{session_created}'],
                      capture_output=True, text=True, check=True).stdout.split()
sup = int(pane[0])
assert pane[1] == '0', 'pane dead'
c = cmd(sup)
print('SUPERVISOR', sup, 'session_created', pane[2], 'cmd', c, flush=True)
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
print('CHILDREN', json.dumps(children), flush=True)
train = {p for p, cm in children.items() if f'scripts.moe_scale_study train --base {base}' in cm}
assert len(train) == 8 and set(children) == train, 'children mismatch'
q = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout
gpu_pids = {int(l.split(',')[0]) for l in q.strip().splitlines() if l.strip()}
print('GPU_PIDS', sorted(gpu_pids), flush=True)
assert gpu_pids == train, 'GPU owners are not exactly the eight train workers'
os.kill(sup, signal.SIGTERM)
print('SIGTERM_SENT', sup, time.time(), flush=True)
for _ in range(44):
    if (base/'terminal.json').exists():
        break
    time.sleep(5)
for n in ('failure.json', 'terminal.json'):
    print(n, (base/n).read_text() if (base/n).exists() else 'absent', flush=True)
nv = base/'native-verified.json'
if nv.exists():
    print('NATIVE_VERIFIED', hashlib.sha256(nv.read_bytes()).hexdigest(), json.loads(nv.read_text())['samples'][-1]['workers'], flush=True)
else:
    print('NATIVE_VERIFIED absent', flush=True)
PY
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
