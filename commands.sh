#1 +30+a
#th2-6y3o-a2-corpora-cpu-20261029-a02
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PY'
import os,signal,time,json,socket,fcntl,hashlib
from pathlib import Path
from datetime import datetime,timezone
assert socket.gethostname()=='thiennh-p6-6y3o-worker-0'
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-a2-corpora-20261029-a01')
new=root.with_name('6y3o-a2-corpora-20261029-a02')
data=Path('/mnt/local/_data/deep-llms_th2/a2_corpora_30b_20261029')
assert not new.exists()
plan=Path('resources/a2_corpora_30b_20261029.json')
assert hashlib.sha256(plan.read_bytes()).hexdigest()==json.loads((root/'source-manifest.json').read_text())['plan_sha256']
assert json.loads((data/'status.json').read_text())['corpora']=={}
assert not list((data/'parts').glob('*/*/complete.json'))
assert not (data/'failure.json').exists()
def identity(pid):
 p=Path('/proc')/str(pid)
 try:
  fields=(p/'stat').read_text().rsplit(')',1)[1].split()
  return dict(pid=pid,state=fields[0],ppid=int(fields[1]),start=fields[19],cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())
 except FileNotFoundError:return None
parent=identity(677159)
assert parent and str(root/'source/scripts/a2_prepare_corpora.py') in parent['cmd']
assert '--plan '+str(root/'source/plan.json') in parent['cmd']
children=[]
for p in Path('/proc').iterdir():
 if p.name.isdigit():
  item=identity(int(p.name))
  if item and item['ppid']==parent['pid']:children.append(item)
assert len(children)==33,len(children)
assert sum('multiprocessing.spawn' in x['cmd'] for x in children)==32
assert sum('multiprocessing.resource_tracker' in x['cmd'] for x in children)==1
assert all('/mnt/local/conda-py311/envs/a2_env/bin/python3.11' in x['cmd'] for x in children)
owned=[parent]+children
for item in owned:
 now=identity(item['pid'])
 assert now and now['start']==item['start'] and now['cmd']==item['cmd']
 fd=os.pidfd_open(item['pid'])
 try:
  now=identity(item['pid']);assert now and now['start']==item['start']
  signal.pidfd_send_signal(fd,signal.SIGTERM)
 finally:os.close(fd)
deadline=time.monotonic()+30
while True:
 live=[x['pid'] for x in owned if (cur:=identity(x['pid'])) and cur['start']==x['start'] and cur['state']!='Z']
 if not live:break
 if time.monotonic()>deadline:raise RuntimeError('CPU workers did not exit; refuse replacement '+str(live))
 time.sleep(.5)
with (data/'.prepare.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
receipt=dict(utc=datetime.now(timezone.utc).isoformat(),reason='Controller destination-directory resolution repair; no tokenization had started',parent=parent,children=children,all_owned_workers_exited=True,preparation_lock_released=True,downloads_untouched=True)
(root/'cpu-recovery-handoff.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('CPU_WAITING_WORKERS_RELEASED',len(owned),flush=True)
PY
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u -m scripts.launch_a2_corpora \
 --plan /mnt/local/deep-llms_th2/resources/a2_corpora_30b_20261029.json \
 --output /mnt/local/_outputs/deep-llms_th2/6y3o-a2-corpora-20261029-a02 \
 --host thiennh-p6-6y3o-worker-0 --session a2_corpora_20261029_a02 \
 --code-commit 5fdf313eb2fb05f239e056b543266b0c69610f1d
