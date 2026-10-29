#1 +30+a
#th2-6y3o-a2-stage1b-fwe-shuffle-health-20261030-a02
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PY'
import json,hashlib,socket,subprocess
from pathlib import Path
from datetime import datetime,timezone
assert socket.gethostname()=='thiennh-p6-6y3o-worker-0'
r=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-a2-stage1b-fwe-shuffle-20261030-a01');d=Path('/mnt/local/_data/deep-llms_th2/a2_stage1b_fwe_shuffle2030_20261030')
print('HEALTH_AT',datetime.now(timezone.utc).isoformat(),flush=True)
for n,p in [('source-manifest',r/'source-manifest.json'),*[(n,d/(n+'.json')) for n in ('status','manifest','complete','diagnostics')]]:
 if p.is_file():
  b=p.read_bytes();print('ARTIFACT',json.dumps(dict(path=n+'.json',sha256=hashlib.sha256(b).hexdigest(),text=b.decode())),flush=True)
 else: print('MISSING',n,flush=True)
sm=json.loads((r/'source-manifest.json').read_text())
for n,h in sm['files'].items():assert hashlib.sha256((r/'source'/n).read_bytes()).hexdigest()==h
print('SOURCE_VERIFIED',sm['code_commit'],len(sm['files']),flush=True)
p=r/'prepare.log'
if p.is_file():
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-6000));print('LOG_TAIL',f.read().decode(errors='replace'),flush=True)
print('SESSION',subprocess.run(['tmux','display-message','-p','-t','a2_stage1b_fwe_shuffle_20261030_a01','#{pane_dead} #{pane_dead_status} #{pane_pid}'],capture_output=True,text=True).stdout.strip(),flush=True)
print('OUTPUT_SIZES',json.dumps({p.name:p.stat().st_size for p in d.glob('*') if p.is_file()}),flush=True)
from scripts.gpu_process import inspect
print('GPU_INSPECTION',json.dumps(inspect()),flush=True)
print('HEALTH_COMPLETE',flush=True)
PY
