#1 +60+a
#th2-q359-proxy-resume20k-status-20261011-a03
set -euo pipefail
cd /mnt/local/deep-llms_th2
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'PYREMOTE'
import hashlib,json,shutil,socket,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
from scripts.gpu_status import snapshot
assert socket.gethostname()=='thiennh-p6-q359-worker-0'
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-proxy-resume20k-seed1042-20261011-a01')
run=root/'supervised/run'
def artifact(p):
    if p.is_file():
        raw=p.read_bytes();print('ARTIFACT',json.dumps(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode())),flush=True)
def tail(p,n=18):
    if p.is_file():
        with p.open('rb') as f:
            f.seek(max(0,p.stat().st_size-100000)); data=f.read().decode(errors='replace')
        print('TAIL',str(p),'\n'+'\n'.join(data.splitlines()[-n:]),flush=True)
for rel in ('preflight.json','supervised/run/validate-A.json','supervised/run/validate-P6-iso.json','supervised/run/continuation-summary.json','supervised/run/complete.json'):
    artifact(root/rel)
for arm in ('A','P6-iso'):
    folder=run/'seed-1042'/arm
    artifact(folder/'result.json')
    points=sorted(folder.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
    print('CHECKPOINTS',arm,[p.name for p in points],flush=True)
    if points and (points[-1]/'trainer_state.json').is_file():
        state=json.loads((points[-1]/'trainer_state.json').read_text())
        print('SAVED_STATE',arm,json.dumps(dict(global_step=state['global_step'],max_steps=state['max_steps'],last_logs=state['log_history'][-3:])),flush=True)
for sample in range(2):
    print('AT',datetime.now(timezone.utc).isoformat(),flush=True)
    for rel in ('supervised/supervisor.json','supervised/run/run.json','supervised/burn-verified.json'):artifact(root/rel)
    tail(root/'supervisor.log');tail(root/'supervised/burn.log',8)
    for p in sorted(run.glob('*.log'),key=lambda p:p.stat().st_mtime)[-3:]:tail(p)
    print('GPUS',json.dumps(snapshot(list(range(8)))),flush=True)
    guard=Path('/mnt/local/_gpu_guard/DISABLED')
    print('GUARD',guard.read_text() if guard.exists() else 'released',flush=True)
    if sample==0:time.sleep(15)
print('FREE_TIB',shutil.disk_usage(root).free/1024**4,flush=True)
print('TMUX',subprocess.run(['tmux','list-panes','-t',root.name,'-F','#{pane_dead} #{pane_pid}'],capture_output=True,text=True).stdout,flush=True)
PYREMOTE
