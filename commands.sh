#1 +30+a
#th2-6y3o-a2-stage1-health-20261029-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PY'
import json,hashlib,subprocess,socket
from pathlib import Path
from datetime import datetime,timezone
assert socket.gethostname()=='thiennh-p6-6y3o-worker-0'
print('HEALTH_AT',datetime.now(timezone.utc).isoformat(),flush=True)
r=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-a2-stage1-shuffle-20261029-a01');d=Path('/mnt/local/_data/deep-llms_th2/a2_stage1_dclm_shuffle2030_20261029')
for label,p in [('source-manifest.json',r/'source-manifest.json'),*[(f'data/{n}',d/n) for n in ('status.json','manifest.json','complete.json')],('corpora/status.json',Path('/mnt/local/_data/deep-llms_th2/a2_corpora_30b_20261029/status.json'))]:
 if p.is_file():
  b=p.read_bytes();print('ARTIFACT',json.dumps(dict(path=label,sha256=hashlib.sha256(b).hexdigest(),text=b.decode())),flush=True)
 else:print('MISSING',label,flush=True)
p=r/'prepare.log'
if p.is_file():
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-6000));print('LOG_TAIL',f.read().decode(errors='replace'),flush=True)
print('SESSION',subprocess.run(['tmux','display-message','-p','-t','a2_stage1_shuffle_20261029_a01','#{pane_dead} #{pane_dead_status} #{pane_pid}'],capture_output=True,text=True).stdout,flush=True)
print('GPU_INSPECTION',flush=True)
from scripts.gpu_process import inspect
print(json.dumps(inspect()),flush=True)
print('HEALTH_COMPLETE',flush=True)
PY
