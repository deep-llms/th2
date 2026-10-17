#1 +60+a
#th2-q359-template-realdata-correct-schedule-20261017-a01
set -euo pipefail
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/eval_fa4/bin/python -B - <<'PY'
import json,signal,os,time
from pathlib import Path
from scripts.verified_gpu_reclaim import process,pidfd_open,pidfd_send_signal,inspect
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-template-realdata-20261017-a01/supervised')
guard=Path('/mnt/local/_gpu_guard/DISABLED')
if guard.exists():
 marker=guard.read_text();owner=json.loads(marker)
 assert owner['output']==str(root),'Different GPU owner; refuse'
 pid=owner['owner_pid'];before=process(pid)
 argv=[x.decode() for x in (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0') if x]
 assert 'scripts.train_then_burn' in argv and str(root) in argv,argv
 fd=pidfd_open(pid)
 try:
  assert process(pid)==before and guard.read_text()==marker,'Owner identity changed'
  print('STOP_OWN_TEMPLATE_PILOT_SUPERVISOR',pid,before['start_ticks'],flush=True)
  pidfd_send_signal(fd,signal.SIGTERM)
 finally:os.close(fd)
else:print('NO_ACTIVE_GUARD',flush=True)
for _ in range(60):
 p=root/'burn-verified.json'
 if not guard.exists() and p.exists():
  receipt=json.loads(p.read_text());now=inspect()
  assert receipt['collective_progress_verified'] and [g['pids'] for g in receipt['gpus']]==[g['pids'] for g in now['gpus']]
  print('PREVIOUS_PILOT_BURN_HANDOFF_VERIFIED',json.dumps(now),flush=True)
  print('SUPERVISOR', (root/'supervisor.json').read_text(),flush=True)
  break
 time.sleep(5)
else:raise RuntimeError('Handoff not yet verified; do not launch a competing job')
PY
