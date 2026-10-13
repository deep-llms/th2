#1 +15+a
#th2-q359-screen22-health-20261013-a03
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'PYREMOTE'
import json,socket,subprocess
from datetime import datetime,timezone
from pathlib import Path
from scripts.gpu_status import snapshot
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-proxy-12layer-screen22-20261013-a01');run=root/'supervised/run'
print('AT',datetime.now(timezone.utc).isoformat(),socket.gethostname(),flush=True)
def tail(p,n=25):
 if not p.is_file():return
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-50000));lines=f.read().decode(errors='replace').splitlines()
 print('TAIL',str(p),'\n'+'\n'.join(lines[-n:]),flush=True)
for name in ('supervisor.json','reclaim.json','gpus-free-before-training.json','burn-verified.json'):
 p=root/'supervised'/name
 if p.exists():print('RECEIPT',name,p.read_text(),flush=True)
for p in (run/'run.json',run/'complete.json'):
 if p.exists():print('QUEUE',p.name,p.read_text(),flush=True)
tail(root/'supervisor.log')
for p in sorted((run/'numerics').glob('case-*.json')):
 x=json.loads(p.read_text());print('NUMERIC',p.name,json.dumps({**{k:x.get(k) for k in ('status','arm','losses','peak_allocated_gib')},'gradient_relative_l2':x['comparison']['gradient_relative_l2'],'proxy_gradient_relative_l2':x['proxy_comparison']['gradient_relative_l2']}),flush=True)
for p in sorted((run/'numerics').glob('case-*.log')):
 if not p.with_suffix('.json').exists():tail(p,12)
for p in sorted(run.glob('*.log'),key=lambda p:p.stat().st_mtime)[-3:]:tail(p,30)
for sub in ('smoke/seed-42','seed-42'):
 for p in sorted((run/sub).glob('*/result.json')):
  x=json.loads(p.read_text());print('RESULT',sub,x['arm'],x['global_step'],x['status'],x['evaluation']['eval_lm_loss'],flush=True)
for d in sorted((run/'seed-42').glob('*')):
 if not d.is_dir():continue
 checkpoints=sorted(d.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
 print('CHECKPOINTS',d.name,[p.name for p in checkpoints],flush=True)
 if checkpoints:
  p=checkpoints[-1]/'trainer_state.json'
  if p.is_file():
   x=json.loads(p.read_text());print('TRAINER_STATE',d.name,json.dumps({k:x.get(k) for k in ('global_step','epoch','max_steps')}),'RECENT',json.dumps(x.get('log_history',[])[-5:]),flush=True)
print('GPU',json.dumps(snapshot(list(range(8)))),flush=True)
print('TMUX',subprocess.run(['tmux','list-panes','-t',root.name,'-F','#{pane_dead} #{pane_pid}'],capture_output=True,text=True).stdout,flush=True)
PYREMOTE
