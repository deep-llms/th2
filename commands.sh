#1 +180+a
#th2-tjx3-p4-sdpa-stop-for-fa4-20261006-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
/mnt/local/conda-py311/envs/train_env/bin/python3.11 -u - <<'PY'
import json,os,signal,socket,time
from pathlib import Path
from scripts.gpu_status import snapshot
from scripts.verified_gpu_reclaim import process,pidfd_open,pidfd_send_signal
from scripts.train_then_burn import argv
from run_experiments import write_json,now
assert socket.gethostname()=='thiennh-p6-tjx3-worker-0'
root=Path('/mnt/local/_outputs/deep-llms_th2/p4-four-head-2500-20261006-a01')
control=Path('/mnt/local/_outputs/deep-llms_th2/p4-sdpa-stop-for-fa4-20261006-a01');control.mkdir(exist_ok=False)
state=json.loads((root/'supervised/supervisor.json').read_text())
print('BEFORE',json.dumps(snapshot(list(range(8)))),json.dumps(state),flush=True)
if not state.get('finished_at'):
 guard=Path('/mnt/local/_gpu_guard/DISABLED');marker=guard.read_text();owner=json.loads(marker)
 assert owner['output']==str(root/'supervised')
 pid=owner['owner_pid'];assert pid>1
 record=process(pid);args=argv(pid)
 assert args[1:4]==['-u','-m','scripts.train_then_burn'],args
 assert args[args.index('--output')+1]==str(root/'supervised')
 assert record['exe']=='/mnt/local/conda-py311/envs/train_env/bin/python3.11',record
 fd=pidfd_open(pid)
 try:
  assert process(pid)==record and guard.read_text()==marker
  write_json(control/'stop-request.json',dict(at=now(),supervisor=record,argv=args,reason='User selected FA4; preserve SDPA outputs'))
  pidfd_send_signal(fd,signal.SIGTERM)
 finally:os.close(fd)
 print('SIGTERM_SENT_TO_VERIFIED_SUPERVISOR',pid,flush=True)
for _ in range(30):
 state=json.loads((root/'supervised/supervisor.json').read_text())
 if state.get('finished_at'):break
 time.sleep(5)
assert state.get('finished_at') and state.get('burn',{}).get('collective_progress_verified'),state
assert not Path('/mnt/local/_gpu_guard/DISABLED').exists()
write_json(control/'handoff.json',state)
print('VERIFIED_TERMINAL_HANDOFF',json.dumps(state),flush=True)
print('AFTER',json.dumps(snapshot(list(range(8)))),flush=True)
PY
