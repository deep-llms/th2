#1 +60+a
#th2-q359-p6-workflow-health-20261011-a01
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
        f.seek(max(0,p.stat().st_size-60000));data=f.read().decode(errors='replace')
    print('TAIL',str(p),'\n'+'\n'.join(data.splitlines()[-n:]),flush=True)
def state(p):
    if p.is_file():
        raw=p.read_bytes()
        print('ARTIFACT',json.dumps({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'text':raw.decode()}),flush=True)
for sample in range(2):
    print('AT',datetime.now(timezone.utc).isoformat(),flush=True)
    for name in ('q359-p6-sync-20261011-a01','q359-p6-cooldown-20261011-a01'):
        root=base/name;run=root/'supervised/run'
        tail(root/'supervisor.log',18);tail(root/'cpu-tests.log',8)
        for rel in ('supervised/supervisor.json','supervised/burn-verified.json','supervised/run/run.json','supervised/run/summary.json','supervised/run/audit/verified.json','supervised/run/branches/plan.json','supervised/run/branches/smoke-verified.json','supervised/run/branches/time-budget.json','supervised/run/branches/summary.json'):
            state(root/rel)
        for p in sorted((run/'branches').glob('*/result.json')):state(p)
        for p in sorted(run.glob('*.log'),key=lambda p:p.stat().st_mtime)[-3:]:tail(p,14)
        tail(root/'supervised/burn.log',6)
        print('TMUX',name,subprocess.run(['tmux','list-panes','-t',name,'-F','#{pane_dead} #{pane_pid}'],capture_output=True,text=True).stdout,flush=True)
    print('GPUS',json.dumps(snapshot(list(range(8)))),flush=True)
    if sample==0:time.sleep(15)
PYREMOTE
