#1 +60+a
#th2-q359-seed42-handoff-check-20261011-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
/mnt/local/conda-py311/envs/eval_fa4/bin/python3.11 -u - <<'PYREMOTE'
import hashlib,json,socket,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
from scripts.gpu_status import snapshot
assert socket.gethostname()=='thiennh-p6-q359-worker-0'
base=Path('/mnt/local/_outputs/deep-llms_th2')
def tail(p,n=12):
    if not p.is_file():return
    with p.open('rb') as f:
        f.seek(max(0,p.stat().st_size-80000));data=f.read().decode(errors='replace')
    print('TAIL',str(p),'\n'+'\n'.join(data.splitlines()[-n:]),flush=True)
for sample in range(2):
    print('AT',datetime.now(timezone.utc).isoformat(),flush=True)
    for name in ('q359-p6-cooldown-20261011-a01','q359-proxy-10k-seed42-20261011-a01'):
        root=base/name;run=root/'supervised/run'
        tail(root/'supervisor.log',20);tail(root/'cpu-tests.log',7)
        for rel in ('preflight.json','supervised/supervisor.json','supervised/run/run.json','supervised/burn-verified.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/run/branches/summary.json'):
            p=root/rel
            if p.is_file():
                raw=p.read_bytes()
                print('ARTIFACT',json.dumps(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode())),flush=True)
        for p in sorted(run.glob('*.log'),key=lambda p:p.stat().st_mtime)[-2:]:tail(p,8)
        print('TMUX',name,subprocess.run(['tmux','list-panes','-t',name,'-F','#{pane_dead} #{pane_pid}'],capture_output=True,text=True).stdout,flush=True)
    guard=Path('/mnt/local/_gpu_guard/DISABLED')
    print('GUARD',guard.read_text() if guard.exists() else 'released',flush=True)
    print('GPUS',json.dumps(snapshot(list(range(8)))),flush=True)
    if sample==0:time.sleep(15)
for name in ('scripts/launch_proxy_10k_fresh.sh','scripts/proxy_replication_queue.py','scripts/train_then_burn.py','resources/accelerate_config.yaml'):
    print('SOURCE_SHA256',name,hashlib.sha256(Path(name).read_bytes()).hexdigest(),flush=True)
PYREMOTE
