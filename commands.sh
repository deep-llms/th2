#1 +45+a
#th2-pq5r-gpu-owner-inspect-20261128-a01
set -euo pipefail
printf 'READ_ONLY_GPU_OWNER_INSPECTION (no process is signaled)\n'
date -u
hostname
TASK_PY=/mnt/local/conda-py311/envs/a2_env/bin/python3.11
TASK_OUT=/mnt/local/_outputs/deep-llms_th2/dot_l34_20261128
mkdir -p "$TASK_OUT"
test ! -e "$TASK_OUT/inspection-a01.json"
"$TASK_PY" - "$TASK_OUT/inspection-a01.json" <<'PYINSPECT'
import hashlib, json, os, sys
from pathlib import Path
from scripts.gpu_process import inspect
state = inspect()
def cmd(pid):
    raw = (Path('/proc') / str(pid) / 'cmdline').read_bytes()
    return [x.decode(errors='replace')[:160] for x in raw.split(b'\0') if x][:6]
def stat(pid):
    f = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
    return int(f[1]), int(f[19])
chains = {}
for pid in state['workers']:
    chain, p = [], pid
    while p > 1 and len(chain) < 8:
        ppid, start = stat(p)
        chain.append(dict(pid=p, ppid=ppid, start_ticks=start, argv=cmd(p), exe=os.readlink(f'/proc/{p}/exe')))
        p = ppid
    chains[str(pid)] = chain
state['ancestor_chains'] = chains
files = {}
for path in ['/tmp/llm_pretrain_burn.py', '/mnt/local/_gpu_guard/polite_burn.py', '/mnt/local/_gpu_guard/gpu_guard.sh',
             'resources/llm_pretrain_burn.py']:
    p = Path(path)
    files[path] = dict(exists=p.is_file(), sha256=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None,
                       bytes=p.stat().st_size if p.is_file() else None)
state['files'] = files
guard = Path('/mnt/local/_gpu_guard')
state['guard_dir'] = sorted(x.name for x in guard.iterdir()) if guard.is_dir() else None
Path(sys.argv[1]).write_text(json.dumps(state, indent=1) + '\n')
print(json.dumps(dict(host=state['host'], workers=state['workers'], stop_roots=state['stop_roots'],
                      guard_disabled=state['guard_disabled'], guard_dir=state['guard_dir'], files=files), indent=1))
for pid, chain in chains.items():
    print('CHAIN', pid, ' <- '.join(f"{c['pid']}:{' '.join(c['argv'][:3])}" for c in chain))
PYINSPECT
sha256sum "$TASK_OUT/inspection-a01.json"
printf 'TMUX\n'; tmux ls 2>&1 || true
printf 'PID1 %s\n' "$(tr '\0' ' ' </proc/1/cmdline | cut -c1-200)"
ls -la /tmp/*.py 2>&1 | head
crontab -l 2>&1 | head -5 || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
df -h /mnt/local | tail -1
printf 'INSPECTION_DONE\n'
