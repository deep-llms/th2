#1 +30+a
#th2-6y3o-a2-stage1-downstream-health-20261030-a03
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PY'
import hashlib,json,socket,subprocess
from pathlib import Path
from datetime import datetime,timezone
from scripts.gpu_process import inspect
assert socket.gethostname()=='thiennh-p6-6y3o-worker-0'
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-a2-stage1-downstream-20261029-a01')
print('HEALTH_AT',datetime.now(timezone.utc).isoformat(),flush=True)
print('GPU_INSPECTION',json.dumps(inspect()),flush=True)
for rel in ('source-manifest.json','plan.json','jobs.json','stage-budget.json','accelerate-env.log','identities/identities.json',
 'supervised/supervisor.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/burn-verified.json',
 'supervised/run/run.json','supervised/run/complete.json','supervised/run/downstream/complete.json','supervised/run/downstream/results.json','supervised/run/downstream/paired_accuracy.json',
 'supervised/run/downstream/P300-parent-check.json','supervised/run/downstream/P300-student-check.json'):
 p=root/rel
 if p.is_file():
  raw=p.read_bytes();assert len(raw)<5000000
  print('ARTIFACT',json.dumps(dict(path=rel,sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode())),flush=True)
 else:print('MISSING',rel,flush=True)
for p in sorted((root/'identities').glob('*-identity.json')):
 raw=p.read_bytes();print('ARTIFACT',json.dumps(dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode())),flush=True)
base=root/'supervised/run/downstream'
p=base/'prepared.json'
if p.is_file():
 raw=p.read_bytes();obj=json.loads(raw)
 print('PREPARED_SUMMARY',json.dumps(dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),requests=len(obj['requests']),protocol=obj['protocol'])),flush=True)
for p in sorted((base/'scores').glob('*/scores.json')):
 raw=p.read_bytes();obj=json.loads(raw)
 print('SCORED_SUMMARY',json.dumps(dict(name=p.parent.name,sha256=hashlib.sha256(raw).hexdigest(),requests=len(obj['scores']),identity=obj['checkpoint_identity_sha256'],precision=obj['precision'],batch_size=obj.get('batch_size'))),flush=True)
logs=[root/'supervisor.log',root/'supervised/burn.log',root/'supervised/run/stage1-downstream.log',*base.glob('*.log'),*(base/'scores').glob('*/*.log')]
for p in logs:
 if p.is_file():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-6000));tail=f.read().decode(errors='replace')
  print('LOG_TAIL',json.dumps(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,text=tail)),flush=True)
guard=Path('/mnt/local/_gpu_guard/DISABLED');print('GUARD',guard.read_text() if guard.is_file() else 'released',flush=True)
print('SESSION',subprocess.run(['tmux','display-message','-p','-t','a2_stage1_downstream_20261029_a01','#{pane_dead} #{pane_pid}'],capture_output=True,text=True).stdout.strip(),flush=True)
print('HEALTH_COMPLETE',flush=True)
PY
