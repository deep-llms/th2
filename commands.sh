#1 +60+a
#th2-q359-10k-health-20261010-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'PYREMOTE'
import json, socket, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
from scripts.gpu_status import snapshot
assert socket.gethostname() == 'thiennh-p6-q359-worker-0'
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-proxy-10k-seed1042-20261009-a01')
run=root/'supervised/run'
def tail(p,n=35):
    if not p.is_file(): return
    with p.open('rb') as f:
        f.seek(max(0,p.stat().st_size-150000))
        lines=f.read().decode(errors='replace').splitlines()
    print('TAIL',str(p),'mtime',datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat(),'\n'+'\n'.join(lines[-n:]),flush=True)
print('AT',datetime.now(timezone.utc).isoformat(),socket.gethostname(),flush=True)
for p in (root/'supervised/supervisor.json',run/'run.json',root/'supervised/burn-verified.json'):
    if p.is_file(): print('RECEIPT',str(p),p.read_text(),flush=True)
tail(root/'supervisor.log')
for p in sorted(run.glob('*.log')): tail(p,45)
for arm in ('A','P6-iso'):
    d=run/'seed-1042'/arm
    checkpoints=sorted(d.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
    print('ARM',arm,'CHECKPOINTS',[p.name for p in checkpoints],flush=True)
    for p in (d/'result.json',d/'train_results.json',d/'eval_results.json'):
        if p.is_file(): print(p.name,p.read_text(),flush=True)
    if checkpoints:
        p=checkpoints[-1]/'trainer_state.json'
        if p.is_file():
            s=json.loads(p.read_text()); print('TRAINER_STATE',arm,json.dumps({k:s.get(k) for k in ('global_step','epoch','max_steps')}),'RECENT_METRICS',json.dumps(s.get('log_history',[])[-12:]),flush=True)
print('TMUX',subprocess.run(['tmux','list-panes','-t',root.name,'-F','#{pane_dead} #{pane_pid}'],capture_output=True,text=True).stdout,flush=True)
print('GPU_SAMPLE_1',json.dumps(snapshot(list(range(8)))),flush=True)
print('DISK',subprocess.run(['df','-h',str(root)],capture_output=True,text=True).stdout,flush=True)
time.sleep(15)
print('AT_2',datetime.now(timezone.utc).isoformat(),flush=True)
for p in sorted(run.glob('*.log')): tail(p,10)
print('GPU_SAMPLE_2',json.dumps(snapshot(list(range(8)))),flush=True)
PYREMOTE
