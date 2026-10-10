#1 +60+a
#th2-q359-p6-compile-health-20261011-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
/mnt/local/conda-py311/envs/eval_fa4/bin/python3.11 -u - <<'PYREMOTE'
import json,socket,subprocess,time,hashlib
from datetime import datetime,timezone
from pathlib import Path
from scripts.gpu_status import snapshot
assert socket.gethostname()=='thiennh-p6-q359-worker-0'
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-p6-compile-20261011-a01');run=root/'supervised/run'
def tail(path,n=12):
    if not path.is_file():return
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-50000));data=f.read().decode(errors='replace')
    print('TAIL',str(path),'\n'+'\n'.join(data.splitlines()[-n:]),flush=True)
for sample in range(2):
    print('AT',datetime.now(timezone.utc).isoformat(),flush=True)
    tail(root/'supervisor.log',25);tail(root/'cpu-tests.log',12)
    for p in (root/'preflight.json',root/'supervised/supervisor.json',root/'supervised/gpus-free-before-training.json',run/'run.json',root/'supervised/burn-verified.json',run/'summary.json'):
        if p.is_file():print('STATE',str(p),p.read_text(),flush=True)
    for p in sorted((run/'numerics').glob('rank-*.json')):
        r=json.loads(p.read_text());print('NUMERIC',p.name,r['status'],r['mode'],r['sequence_length'],r['checkpointing'],flush=True)
        for name,c in r['comparisons'].items():
            print('NUMERIC_COMPONENT',name,'differences',len(c['differences']),'failures',len(c['failures']),
                  'max_relative_l2',max((d.get('relative_l2',0) for d in c['differences']),default=0),'example_failures',c['failures'][:3],flush=True)
    p=run/'numerics/summary.json'
    if p.is_file():
        r=json.loads(p.read_text());print('NUMERIC_SUMMARY',{k:v for k,v in r.items() if k!='reports'},flush=True)
    tail(root/'supervised/burn.log',8)
    for p in sorted(run.glob('*.log'),key=lambda p:p.stat().st_mtime)[-3:]:tail(p,22)
    print('GPUS',json.dumps(snapshot(list(range(8)))),flush=True)
    print('TMUX',subprocess.run(['tmux','list-panes','-t',root.name,'-F','#{pane_dead} #{pane_pid}'],capture_output=True,text=True).stdout,flush=True)
    if sample==0:time.sleep(15)
PYREMOTE
