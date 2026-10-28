#1 +30+a
#th2-6y3o-a2-remaining-health-20261028-a05
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PY'
import hashlib,json,socket,subprocess
from pathlib import Path
from datetime import datetime,timezone
from scripts.gpu_process import inspect
assert socket.gethostname()=='thiennh-p6-6y3o-worker-0'
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-a2-remaining-phase0a-20261028-a01')
print('HEALTH_AT',datetime.now(timezone.utc).isoformat(),flush=True)
print('GPU_INSPECTION',json.dumps(inspect()),flush=True)
for rel in ('source-manifest.json', 'inputs-verified.json', 'accelerate-env.log', 'jobs.json', 'history.json', 'reference.json', 'phase-budget.json', 'remaining-plan.json', 'prerequisites.json', 'supervised/supervisor.json', 'supervised/reclaim.json', 'supervised/gpus-free-before-training.json', 'supervised/burn-verified.json', 'supervised/run/run.json', 'supervised/run/complete.json', 'supervised/run/remaining/prelaunch-report.json', 'supervised/run/remaining/baselines.json', 'supervised/run/remaining/heals/reuse-plan.json', 'supervised/run/remaining/heals/results.json', 'supervised/run/remaining/heals/complete.json') + tuple(str(p.relative_to(root)) for p in sorted((root/"supervised/run/remaining").glob("*-pre.json"))):
 p=root/rel
 if p.is_file():
  raw=p.read_bytes();assert len(raw)<5000000
  print('ARTIFACT',json.dumps(dict(path=rel,sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode())),flush=True)
 else:print('MISSING',rel,flush=True)
for rel in ('supervisor.log', 'supervised/burn.log', 'supervised/run/remaining-phase0a.log') + tuple(str(p.relative_to(root)) for p in sorted((root/"supervised/run/remaining").glob("*-pre.log"))) + tuple(str(p.relative_to(root)) for p in sorted((root/"supervised/run/remaining/heals").glob("*.log"))):
 p=root/rel
 if p.is_file():
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-12000));tail=f.read().decode(errors='replace')
  print('LOG_TAIL',json.dumps(dict(path=rel,bytes=p.stat().st_size,text=tail)),flush=True)
guard=Path('/mnt/local/_gpu_guard/DISABLED');print('GUARD',guard.read_text() if guard.is_file() else 'released',flush=True)
print('SESSION',subprocess.run(['tmux','display-message','-p','-t','a2_remaining_phase0a_20261028_a01','#{pane_dead} #{pane_pid}'],capture_output=True,text=True).stdout.strip(),flush=True)
print('HEALTH_COMPLETE',flush=True)
PY
