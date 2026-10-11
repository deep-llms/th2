#1 +60+a
#th2-q359-proxy-resume20k-health-20261011-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'PYREMOTE'
import hashlib,json,socket,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
from scripts.gpu_status import snapshot
from accelerate.commands.config.config_args import default_yaml_config_file
assert socket.gethostname()=='thiennh-p6-q359-worker-0'
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-proxy-resume20k-seed1042-20261011-a01')
run=root/'supervised/run'
def artifact(p):
    if p.is_file():
        raw=p.read_bytes();print('ARTIFACT',json.dumps(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode())),flush=True)
    else:print('MISSING',str(p),flush=True)
def tail(p,n=15):
    if p.is_file():
        with p.open('rb') as f:
            f.seek(max(0,p.stat().st_size-40000)); data=f.read().decode(errors='replace')
        print('TAIL',str(p),'\n'+'\n'.join(data.splitlines()[-n:]),flush=True)
for rel in ('preflight.json','jobs.json','supervised/supervisor.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/run/seed-1042/resume_inputs.json'):
    artifact(root/rel)
tail(root/'retention-test.log',8)
cache=Path(default_yaml_config_file)
print('ACCELERATE_CACHE',str(cache),hashlib.sha256(cache.read_bytes()).hexdigest(),flush=True)
assert cache.read_bytes()==Path('resources/accelerate_config.yaml').read_bytes()
for arm in ('A','P6-iso'):
    folder=run/'seed-1042'/arm
    for p in sorted(folder.glob('resume-transition-*.json')):artifact(p)
    if (folder/'train_config.json').exists():artifact(folder/'train_config.json')
    checkpoint=folder/'checkpoint-10000'
    print('STAGED_STATE',arm,[(p.name,p.stat().st_size) for p in sorted(checkpoint.glob('*')) if p.is_file()],flush=True)
for i in range(2):
    print('AT',datetime.now(timezone.utc).isoformat(),flush=True)
    artifact(run/'run.json');tail(root/'supervisor.log',20)
    for p in sorted(run.glob('*.log'),key=lambda p:p.stat().st_mtime)[-2:]:tail(p,16)
    print('GPUS',json.dumps(snapshot(list(range(8)))),flush=True)
    guard=Path('/mnt/local/_gpu_guard/DISABLED')
    print('GUARD',guard.read_text() if guard.exists() else 'released',flush=True)
    if i==0:time.sleep(15)
print('TMUX',subprocess.run(['tmux','list-panes','-t',root.name,'-F','#{pane_dead} #{pane_pid}'],capture_output=True,text=True).stdout,flush=True)
PYREMOTE
