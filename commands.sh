#1 +60+a
#th2-ohmv-warm-downstream-health-20261118-a01
set -euo pipefail
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 - <<'PY'
import json,pathlib,time,hashlib,tarfile
from efficient_llm.warm_order.data import sha
from scripts.gpu_process import inspect,process
b=pathlib.Path('/mnt/local/_outputs/deep-llms_th2/warm_downstream_20261118')
def read(p):
 p=b/p;return json.loads(p.read_text()) if p.exists() else None
status=read('status.json');terminal=read('terminal.json');plan=read('plan.json');pin=read('execution/source.json')
source_errors=[p for p,w in (pin or {}).items() if sha(b/'execution/source'/p)!=w]
models={}
for e in (plan or {}).get('models',[]):
 n=e['name'];models[n]={'complete':read(f'runs/{n}/complete.json'),'replay':read(f'runs/{n}/replay-complete.json')}
 d=read(f'runs/{n}/details.json')
 if d:models[n]['details']={k:v for k,v in d.items() if k!='choices'}
 if terminal:models[n]['actual_checkpoint_hash_ok']=all(sha(p)==w for p,w in e['hashes'].items())
state=inspect();active={}
for j in (status or {}).get('active',[]):
 try:
  p=process(j['pid']);active[j['name']]={'identity':p,'argv_ok':p['command_sha256']==hashlib.sha256(('\0'.join(j['argv'])+'\0').encode()).hexdigest(),'parent':process(p['ppid'])}
 except FileNotFoundError:active[j['name']]={'exited_since_status':True}
proof=read('backup.json');backup_verified=None
if terminal and proof:
 with tarfile.open(b/'results.tar.gz') as t:
  members=t.getmembers();assert len(members)==len(proof['files']);assert all(m.isfile() and not m.name.endswith('.pt') for m in members)
  assert all(hashlib.sha256(t.extractfile(m).read()).hexdigest()==proof['files'][m.name] for m in members)
 assert sha(b/'results.tar.gz')==proof['archive_sha256']==sha(proof['node_backup'])
 backup_verified={'bytes':proof['bytes'],'archive_sha256':proof['archive_sha256'],'proof_sha256':sha(b/'backup.json'),'members':len(members),'node_backup':proof['node_backup']}
print('DOWNSTREAM_HEALTH',json.dumps({'utc':time.time(),'submitted':read('execution/submitted.json'),'admission':read('execution/admission.json'),'status':status,'terminal':terminal,'failure':read('failure.json'),'probe':read('gpu-probe.json'),'ownership':state,'active':active,'source_errors':source_errors,'models':models,'report':read('REPORT.json'),'backup_verified':backup_verified,'native':read('native-verified.json'),'guard':pathlib.Path('/mnt/local/_gpu_guard/DISABLED').read_text() if pathlib.Path('/mnt/local/_gpu_guard/DISABLED').exists() else None,'queue_tail':(b/'execution/queue.log').read_text()[-4500:] if (b/'execution/queue.log').exists() else None}))
PY
