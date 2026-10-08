#1 +30+a
#th2-tjx3-10k-prestop-inspection-20261008-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
test "$(hostname)" = thiennh-p6-tjx3-worker-0
/mnt/local/conda-py311/envs/attention_bench/bin/python -u - <<'PY'
import json,subprocess
from pathlib import Path
from scripts.gpu_status import snapshot
from scripts.verified_gpu_reclaim import process
root=Path('/mnt/local/_outputs/deep-llms_th2/proxy-followup-2500-20261008-a01/supervised')
for name in ('supervisor.json','run/run.json','run/complete.json','burn-verified.json'):
 p=root/name
 print('FILE',name,p.is_file(),json.loads(p.read_text()) if p.is_file() and name!='run/run.json' else None,flush=True)
 if p.is_file() and name=='run/run.json':
  x=json.loads(p.read_text());print('QUEUE',x.get('status'),[(j['name'],j['status']) for j in x.get('jobs',[])],flush=True)
session='tjx3-proxy-followup-2500-20261008-a01'
r=subprocess.run(['tmux','display-message','-p','-t',session,'#{pane_pid} #{pane_dead}'],capture_output=True,text=True)
print('TMUX',r.returncode,r.stdout.strip(),r.stderr.strip(),flush=True)
if r.returncode==0:
 pid=int(r.stdout.split()[0])
 for _ in range(3):
  if pid<=1:break
  try:
   x=process(pid);print('SESSION_PROCESS',x,flush=True);pid=x['ppid']
  except Exception as e:print('PROCESS_ERROR',repr(e),flush=True);break
gpus=snapshot(list(range(8)))
print('GPUS',gpus,flush=True)
for pid in sorted({p for g in gpus for p in g['pids']}):
 try:
  x=process(pid);print('GPU_WORKER',x,flush=True)
  if x['ppid']>1:print('GPU_PARENT',process(x['ppid']),flush=True)
 except Exception as e:print('GPU_PROCESS_ERROR',pid,repr(e),flush=True)
print('DONE',flush=True)
PY
