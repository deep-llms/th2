#1 +60+a
#th2-q359-stop-seed42-verify-20261011-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
/mnt/local/conda-py311/envs/eval_fa4/bin/python3.11 -u - <<'PYREMOTE'
import hashlib,json,socket,subprocess,time
from pathlib import Path
from scripts.gpu_status import snapshot
assert socket.gethostname()=='thiennh-p6-q359-worker-0'
base=Path('/mnt/local/_outputs/deep-llms_th2')
old=base/'q359-proxy-10k-seed42-20261011-a01'
for rel in ('q359-stop-seed42-20261011-a01/stop.json','q359-stop-seed42-20261011-a01/stop.log', 'q359-proxy-10k-seed42-20261011-a01/supervised/supervisor.json','q359-proxy-10k-seed42-20261011-a01/supervised/run/run.json','q359-proxy-10k-seed42-20261011-a01/supervised/burn-verified.json'):
    p=base/rel
    if p.is_file():
        raw=p.read_bytes();print('ARTIFACT',json.dumps(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode())),flush=True)
    else:print('MISSING',str(p),flush=True)
for arm in ('A','P6-iso'):
    folder=old/'supervised/run/seed-42'/arm
    print('PRESERVED_CHECKPOINTS',arm,[p.name for p in sorted(folder.glob('checkpoint-*'))],flush=True)
for i in range(2):
    guard=Path('/mnt/local/_gpu_guard/DISABLED')
    print('GUARD',guard.read_text() if guard.exists() else 'released',flush=True)
    print('GPUS',json.dumps(snapshot(list(range(8)))),flush=True)
    log=old/'supervised/final-burn.log'
    if log.exists():print('BURN_TAIL',log.read_text()[-4000:],flush=True)
    if i==0:time.sleep(15)
print(subprocess.run(['tmux','list-panes','-t',old.name,'-F','#{pane_dead} #{pane_pid}'],capture_output=True,text=True).stdout)
PYREMOTE
