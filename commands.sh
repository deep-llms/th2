#1 +60+a
#th2-q359-recurrent-health-20261011-a03
set -euo pipefail
cd /mnt/local/deep-llms_th2
/mnt/local/conda-py311/envs/eval_fa4/bin/python3.11 -u - <<'PYREMOTE'
import json,socket,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
from scripts.gpu_status import snapshot
assert socket.gethostname()=='thiennh-p6-q359-worker-0'
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-recurrent-p6-20261011-a02')
run=root/'supervised/run'
def tail(path,n=16):
    if not path.is_file():return
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-40000));data=f.read().decode(errors='replace')
    print('TAIL',str(path),'\n'+'\n'.join(data.splitlines()[-n:]),flush=True)
if (run/'smoke.log').is_file():
    lines=(run/'smoke.log').read_text().splitlines()
    print('SMOKE_ERRORS', '\n'.join(line for line in lines if '[rank' in line or 'Error:' in line), flush=True)
for sample in range(2):
    print('AT',datetime.now(timezone.utc).isoformat(),flush=True)
    tail(root/'supervisor.log',25);tail(root/'cpu-tests.log',12)
    print('AUDIT_FILES',[p.name for p in root.glob('*validation*.json')]+[p.name for p in root.glob('prompt-audit*.json')],flush=True)
    for p in (root/'preflight.json',root/'supervised/supervisor.json',root/'supervised/gpus-free-before-training.json',run/'run.json',run/'diagnostic/summary.json',root/'supervised/burn-verified.json'):
        if p.is_file():print('STATE',str(p),p.read_text(),flush=True)
    for p in sorted(run.glob('full-*-validation.json')):print('RESULT',str(p),p.read_text(),flush=True)
    tail(root/'supervised/burn.log',8)
    for p in sorted(run.rglob('summary.json')):print('RESULT',str(p),p.read_text(),flush=True)
    for p in sorted(run.rglob('run.json')):
        if p!=run/'run.json':
            r=json.loads(p.read_text());print('NESTED_RUN',str(p),r['status'],'LAST_JOBS',json.dumps(r.get('jobs',[])[-3:]),flush=True)
    for p in sorted(run.rglob('*.log'),key=lambda p:p.stat().st_mtime)[-4:]:tail(p,30)
    print('GPUS',json.dumps(snapshot(list(range(8)))),flush=True)
    print('TMUX',subprocess.run(['tmux','list-panes','-t',root.name,'-F','#{pane_dead} #{pane_pid}'],capture_output=True,text=True).stdout,flush=True)
    if sample==0:time.sleep(15)
PYREMOTE
