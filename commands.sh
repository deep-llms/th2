#1 +30+a
#th2-6y3o-a2-corpora-health-20261029-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PY'
from pathlib import Path
import json,hashlib,socket,subprocess
from datetime import datetime,timezone
assert socket.gethostname()=='thiennh-p6-6y3o-worker-0'
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-a2-corpora-20261029-a01')
data=Path('/mnt/local/_data/deep-llms_th2/a2_corpora_30b_20261029')
print('HEALTH_AT',datetime.now(timezone.utc).isoformat(),flush=True)
for label,p in [('source-manifest.json',root/'source-manifest.json'),('data/status.json',data/'status.json'),('data/failure.json',data/'failure.json'),('data/complete.json',data/'complete.json'),*[(f'data/{c}/manifest.json',data/'packed'/c/'manifest.json') for c in ('dclm','fineweb_edu')]]:
 if p.is_file():
  b=p.read_bytes();print('ARTIFACT',json.dumps(dict(path=label,sha256=hashlib.sha256(b).hexdigest(),text=b.decode())),flush=True)
 else:print('MISSING',label,flush=True)
plan=json.loads((root/'source/plan.json').read_text())
for name,corpus in plan['corpora'].items():
 files=[Path(f['local']) for f in corpus['files']];present=[p for p in files if p.is_file()]
 parts=list((data/'parts'/name).glob('*/complete.json'))
 print('CORPUS_COUNTS',json.dumps(dict(corpus=name,raw_files_present=len(present),raw_bytes_observed=sum(p.stat().st_size for p in present),planned_files=len(files),completed_parts=len(parts))),flush=True)
log=root/'prepare.log'
if log.is_file():
 with log.open('rb') as f:f.seek(max(0,log.stat().st_size-12000));tail=f.read().decode(errors='replace')
 print('LOG_TAIL',json.dumps(dict(path='prepare.log',bytes=log.stat().st_size,text=tail)),flush=True)
print('SESSION',subprocess.run(['tmux','display-message','-p','-t','a2_corpora_20261029_a01','#{pane_dead} #{pane_pid}'],capture_output=True,text=True).stdout.strip(),flush=True)
print('HEALTH_COMPLETE',flush=True)
PY
