#1 +15+a
#th2-q359-screen22-health-20261013-a04
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
def tail(p,n=35):
 if not p.is_file():return
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-120000));lines=f.read().decode(errors='replace').splitlines()
 print('TAIL',str(p),'\n'+'\n'.join(lines[-n:]),flush=True)
for name in ('supervisor.json','burn-verified.json'):
 p=root/'supervised'/name
 if p.exists():print('RECEIPT',name,p.read_text(),flush=True)
for p in (run/'run.json',run/'complete.json'):
 if p.exists():print('QUEUE',p.name,p.read_text(),flush=True)
tail(root/'supervisor.log',10)
for sub in ('smoke/seed-42','seed-42'):
 for d in sorted((run/sub).glob('*')):
  if not d.is_dir():continue
  p=d/'result.json'
  if p.is_file():
   x=json.loads(p.read_text());print('RESULT',sub,json.dumps({k:x.get(k) for k in ('arm','global_step','status','training_cost')}),'LM',x['evaluation']['eval_lm_loss'],flush=True)
  checkpoints=sorted(d.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
  if checkpoints:
   p=checkpoints[-1]/'trainer_state.json'
   if p.is_file():
    x=json.loads(p.read_text());rows=[{k:r.get(k) for k in ('step','loss','step_lm_loss','grad_norm','seconds_per_update')} for r in x.get('log_history',[]) if 'loss' in r]
    print('STATE',sub,d.name,'STEP',x['global_step'],'CHECKPOINTS',len(checkpoints),'RECENT',json.dumps(rows[-5:]),flush=True)
for p in sorted(run.glob('*.log'),key=lambda p:p.stat().st_mtime)[-2:]:tail(p,35)
print('GPU',json.dumps(snapshot(list(range(8)))),flush=True)
print('TMUX',subprocess.run(['tmux','list-panes','-t',root.name,'-F','#{pane_dead} #{pane_pid}'],capture_output=True,text=True).stdout,flush=True)
PYREMOTE
