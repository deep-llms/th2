#1 +30+a
#th2-6y3o-a2-nem-s1-health-20261028-a03
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PY'
import hashlib,json,socket,subprocess
from pathlib import Path
from datetime import datetime,timezone
from scripts.gpu_process import inspect
assert socket.gethostname()=='thiennh-p6-6y3o-worker-0'
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-a2-nem-s1-20261028-a01')
print('HEALTH_AT',datetime.now(timezone.utc).isoformat(),flush=True)
print('GPU_INSPECTION',json.dumps(inspect()),flush=True)
for rel in ('source-manifest.json','inputs-verified.json','accelerate-env.log','jobs.json','supervised/supervisor.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/burn-verified.json','supervised/run/run.json','supervised/run/complete.json','supervised/run/pair/shared-check-reuse.json','supervised/run/pair/P300-survey.json','supervised/run/pair/P1T-survey.json','supervised/run/pair/prelaunch-report.json','supervised/run/pair/gate.json','supervised/run/pair/reuse.json','supervised/run/pair/P300-pre.json','supervised/run/pair/P1T-pre.json','supervised/run/pair/chosen-lrs.json','supervised/run/pair/timings.json','supervised/run/pair/results.json','supervised/run/pair/complete.json'):
 p=root/rel
 if p.is_file():
  raw=p.read_bytes();assert len(raw)<2500000
  print('ARTIFACT',json.dumps(dict(path=rel,sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode())),flush=True)
 else:print('MISSING',rel,flush=True)
for rel in ('supervisor.log','supervised/run/s1-pair.log','supervised/run/pair/P300-survey.log','supervised/run/pair/P1T-survey.log','supervised/run/pair/P300-pre.log','supervised/run/pair/P1T-pre.log','supervised/burn.log'):
 p=root/rel
 if p.is_file():
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-12000));tail=f.read().decode(errors='replace')
  print('LOG_TAIL',json.dumps(dict(path=rel,bytes=p.stat().st_size,text=tail)),flush=True)
for p in sorted((root/'supervised/run/pair').glob('P0-*.log')):
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-10000));tail=f.read().decode(errors='replace')
 print('FIT_LOG_TAIL',json.dumps(dict(path=str(p.relative_to(root)),text=tail)),flush=True)
guard=Path('/mnt/local/_gpu_guard/DISABLED');print('GUARD',guard.read_text() if guard.is_file() else 'released',flush=True)
print('SESSION',subprocess.run(['tmux','display-message','-p','-t','a2_nem_s1_20261028_a01','#{pane_dead} #{pane_pid}'],capture_output=True,text=True).stdout.strip(),flush=True)
print('HEALTH_COMPLETE',flush=True)
PY
