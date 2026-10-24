#1 +60+a
#th2-6y3o-dot-stop-verify-cancel-20261025-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-6y3o-worker-0
/mnt/local/conda-py311/envs/train/bin/python3.11 - <<'REMOTE'
import json, os, signal, time
from pathlib import Path
from scripts.gpu_process import inspect, process, pidfd_open, pidfd_send_signal, ownership
from scripts.native_baseline_pipeline import burn_gate
from scripts.train_then_burn import argv
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-dot-stop-optimized-20261025-a01')
guard=Path('/mnt/local/_gpu_guard/DISABLED')
read=lambda p:json.loads(p.read_text())
state=read(root/'supervised/supervisor.json')
assert state['training_status']=='running', state
marker=guard.read_bytes(); owner=json.loads(marker)
assert owner['output']==str(root/'supervised'), owner
pid=owner['owner_pid']; before=inspect(); identity=process(pid); arguments=argv(pid)
assert identity['exe']=='/mnt/local/conda-py311/envs/train/bin/python3.11'
assert arguments[1:3]==['-m','scripts.train_then_burn'],arguments
assert arguments[arguments.index('--config')+1]==str(root/'jobs.json')
assert arguments[arguments.index('--output')+1]==str(root/'supervised')
assert Path('/proc',str(pid),'cwd').resolve()==root/'source'
assert not any(j['name'].startswith('D-') for j in read(root/'supervised/run/run.json')['jobs']), 'Long training already started; refusing test-only cancellation'
for worker in before['workers']:
 current=worker; visited=set()
 while current!=pid:
  assert current>1 and current not in visited
  visited.add(current); current=process(current)['ppid']
fd=pidfd_open(pid)
try:
 assert process(pid)==identity and guard.read_bytes()==marker
 assert ownership(inspect()['gpus'])==ownership(before['gpus'])
 receipt={'action':'verification fixture mismatch: stop own test queue before long training; preserve all outputs',
          'supervisor':identity,'argv':arguments,'before':before,'queue':read(root/'supervised/run/run.json')}
 with (root/'stop-request-20261025-a01.json').open('x') as f:json.dump(receipt,f,indent=2)
 pidfd_send_signal(fd,signal.SIGTERM)
 print('AUTHORIZED_SUPERVISOR_STOP '+json.dumps(receipt),flush=True)
finally:os.close(fd)
for attempt in range(60):
 time.sleep(5)
 state=read(root/'supervised/supervisor.json')
 if state.get('finished_at'):
  assert not state.get('handoff_error'),state
  burn_gate(root)
  print('OLD_QUEUE_STOP_AND_BURN_HANDOFF_VERIFIED '+json.dumps(state),flush=True)
  break
else:raise RuntimeError('Handoff not yet verified; inspect only, do not launch competing jobs')
REMOTE
