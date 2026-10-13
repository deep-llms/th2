#1 +15+a
#th2-q359-followup-handoff-verify-20261014-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'PYREMOTE'
import hashlib,json,subprocess
from pathlib import Path
from datetime import datetime,timezone
from scripts.proxy_overnight_queue import CASES,predecessor_ready
base=Path('/mnt/local/_outputs/deep-llms_th2')
previous=base/'q359-proxy-12layer-screen22-20261013-a01'
root=base/'q359-proxy-followup11-20261014-a01'
print('AT',datetime.now(timezone.utc).isoformat(),flush=True)
r=subprocess.run(['tmux','list-panes','-t',root.name,'-F','#{pane_dead} #{pane_pid} #{pane_current_command}'],capture_output=True,text=True)
print('WAITER_PANES',r.returncode,r.stdout.strip(),r.stderr.strip(),flush=True)
for folder in (previous,root):
 print('ROOT',folder.name,flush=True)
 for relative in ('supervised/supervisor.json','supervised/run/run.json'):
  p=folder/relative
  if not p.exists():print('MISSING',relative,flush=True);continue
  state=json.loads(p.read_text())
  if 'jobs' in state:
   print('RUN',state['status'],'PASSED',sum(j['status']=='ok' for j in state['jobs']),flush=True)
   print('ACTIVE',[j for j in state['jobs'] if j['status'] not in ('ok','pending','queued')],flush=True)
  else:print('SUPERVISOR',json.dumps(state),flush=True)
 p=folder/'supervisor.log'
 if p.exists():
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-10000));lines=f.read().decode(errors='replace').splitlines()
  print('TAIL','\n'.join(lines[-8:]),flush=True)
print('PREDECESSOR_READY',predecessor_ready(previous),flush=True)
print('FOLLOWUP_CASES',json.dumps(CASES),flush=True)
for p in ('scripts/launch_proxy_12layer.sh','scripts/proxy_overnight_queue.py','scripts/train_then_burn.py','deep_kv/proxy.py','deep_kv/proxy_estimators.py','deep_kv/training.py','resources/accelerate_config.yaml'):
 print('SHA256',p,hashlib.sha256(Path(p).read_bytes()).hexdigest(),flush=True)
if r.returncode==0:
 for line in r.stdout.splitlines():
  dead,pid,*rest=line.split();assert dead=='0','Follow-up pane has exited'
  cmd=(Path('/proc')/pid/'cmdline').read_bytes().replace(b'\0',b' ').decode()
  print('PANE_COMMAND',pid,cmd,flush=True)
 print('FOLLOWUP_ALIVE',flush=True)
else:
 p=root/'supervised/supervisor.json'
 assert p.exists() and json.loads(p.read_text()).get('finished_at'),'Follow-up waiter is not alive'
 print('FOLLOWUP_TERMINAL_INSPECT_RESULT',flush=True)
PYREMOTE
