#1 +60+a
#th2-ohmv-warm-downstream-recovery-health-20261118-a01
set -euo pipefail
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 - <<'PYCODE'
import pathlib,json,time,tarfile,hashlib
from efficient_llm.warm_order.data import sha
from scripts.gpu_process import inspect,ownership
b=pathlib.Path('/mnt/local/_outputs/deep-llms_th2/warm_downstream_replay_20261118_a01')
def read(p):
 q=b/p;return json.loads(q.read_text()) if q.exists() else None
terminal=read('terminal.json');proof=read('backup.json');recovery=read('recovery.json');pin=read('recovery-source.json')
errors=[p for p,w in (pin or {}).items() if sha(b/'recovery-source'/p)!=w]
state=inspect();unchanged=ownership(state['gpus'])==ownership(recovery['ownership']['gpus']) if recovery else False
verified=None
if terminal and proof:
 with tarfile.open(b/'results.tar.gz') as t:
  ms=t.getmembers();assert len(ms)==len(proof['files']);assert all(m.isfile() and not m.name.endswith('.pt') for m in ms)
  assert all(hashlib.sha256(t.extractfile(m).read()).hexdigest()==proof['files'][m.name] for m in ms)
 assert sha(b/'results.tar.gz')==proof['archive_sha256']==sha(proof['node_backup'])
 verified=dict(archive_sha256=proof['archive_sha256'],bytes=proof['bytes'],proof_sha256=sha(b/'backup.json'),proof_bytes=(b/'backup.json').stat().st_size,members=len(ms),node_backup=proof['node_backup'])
print('RECOVERY_HEALTH',json.dumps(dict(utc=time.time(),terminal=terminal,failure=read('failure.json'),source_errors=errors,report=read('REPORT.json'),backup=verified,current_ownership=state,native_unchanged=unchanged,replays={p.parent.name:json.loads(p.read_text()) for p in b.glob('runs/*/replay-complete.json')},logs={str(p.relative_to(b)):p.read_text()[-2500:] for p in [b/'recovery.log',*b.glob('runs/*/replay.log')] if p.exists()})))
PYCODE
