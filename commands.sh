#1 +60+a
#th2-q359-checkpoint-ppl-retry-status-20261012-a04
set -euo pipefail
cd /mnt/local/deep-llms_th2
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'PYREMOTE'
import hashlib,json,socket,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
from scripts.gpu_status import snapshot
from scripts.train_then_burn import burn_progress,approved_launcher,BURN_HASH
from scripts.verified_gpu_reclaim import process
assert socket.gethostname()=='thiennh-p6-q359-worker-0'
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-checkpoint-ppl-20261012-a02');run=root/'supervised/run'
def artifact(p):
    if p.is_file():
        raw=p.read_bytes();print('ARTIFACT',json.dumps(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode())),flush=True)
def tail(p,n=10):
    if p.is_file():
        with p.open('rb') as f:
            f.seek(max(0,p.stat().st_size-40000));s=f.read().decode(errors='replace')
        print('TAIL',str(p),'\n'+'\n'.join(s.splitlines()[-n:]),flush=True)
for rel in ('preflight.json','supervised/gpus-free-before-training.json','supervised/run/summary.json','supervised/run/complete.json'):artifact(root/rel)
for p in sorted(run.glob('*/summary.json')):artifact(p)
tail(root/'cpu-tests.log',5)
for sample in range(2):
    print('AT',datetime.now(timezone.utc).isoformat(),flush=True)
    for rel in ('supervised/supervisor.json','supervised/run/run.json','supervised/burn-verified.json'):artifact(root/rel)
    tail(root/'supervisor.log',18);tail(root/'supervised/burn.log',5)
    for p in sorted(run.glob('*.log'),key=lambda p:p.stat().st_mtime)[-2:]:tail(p,8)
    gpus=snapshot(list(range(8)));print('GPUS',json.dumps(gpus),flush=True)
    if (root/'supervised/burn-verified.json').exists():
        workers=sorted({p for g in gpus for p in g['pids']})
        print('LIVE_BURNS',json.dumps(dict(workers=workers,approved_parents=bool(workers) and all(approved_launcher(process(p)['ppid'],{str(Path.cwd()/'resources/llm_pretrain_burn.py'):BURN_HASH}) for p in workers),collective_progress=burn_progress((root/'supervised/burn.log').read_text()))),flush=True)
    guard=Path('/mnt/local/_gpu_guard/DISABLED');print('GUARD',guard.read_text() if guard.exists() else 'released',flush=True)
    if sample==0:time.sleep(15)
print('TMUX',subprocess.run(['tmux','list-panes','-t',root.name,'-F','#{pane_dead} #{pane_pid}'],capture_output=True,text=True).stdout,flush=True)
PYREMOTE
