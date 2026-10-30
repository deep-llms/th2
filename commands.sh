#1 +30+a
#th2-6y3o-a2-stage1b-fwe-part2-health-20261030-a13
set -euo pipefail
cd /mnt/local/deep-llms_th2
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
/mnt/local/conda-py311/envs/a2_env/bin/python3.11 -u - <<'PY'
import hashlib,json,socket,subprocess
from pathlib import Path
from datetime import datetime,timezone
from scripts.gpu_process import inspect
assert socket.gethostname()=='thiennh-p6-6y3o-worker-0'
root=Path('/mnt/local/_outputs/deep-llms_th2/6y3o-a2-stage1b-fwe-part2-20261030-a01')
print('HEALTH_AT',datetime.now(timezone.utc).isoformat(),flush=True)
print('GPU_INSPECTION',json.dumps(inspect()),flush=True)
rels=['source-manifest.json','inputs.json','stage-budget.json','jobs.json','accelerate-env.log','supervised/supervisor.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/burn-verified.json','supervised/run/run.json','supervised/run/complete.json']
fitroot=root/'supervised/run/results/a2_stage1_fwe'
for name in ('progress.json','GRID_NOTE.md','lr_choice.json','decision_stage1_fwe.json','tests.json','runs.json','complete.json'):rels.append(str((fitroot/name).relative_to(root)))
for p in fitroot.glob('*/result.json'):rels.append(str(p.relative_to(root)))
for rel in rels:
 p=root/rel
 if p.is_file():
  raw=p.read_bytes();assert len(raw)<10000000
  print('ARTIFACT',json.dumps(dict(path=rel,sha256=hashlib.sha256(raw).hexdigest(),text=raw.decode())),flush=True)
 else:print('MISSING',rel,flush=True)
base=root/'supervised/run'
logs=[root/'supervisor.log',root/'supervised/burn.log',*base.glob('*.log'),*fitroot.glob('*.log'),*fitroot.glob('*/profile.txt')]
for p in logs:
 if p.is_file():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-6000));tail=f.read().decode(errors='replace')
  print('LOG_TAIL',json.dumps(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,text=tail)),flush=True)
guard=Path('/mnt/local/_gpu_guard/DISABLED');print('GUARD',guard.read_text() if guard.is_file() else 'released',flush=True)
print('SESSION',subprocess.run(['tmux','display-message','-p','-t','a2_stage1b_fwe_part2_20261030_a01','#{pane_dead} #{pane_pid}'],capture_output=True,text=True).stdout.strip(),flush=True)
processes=subprocess.run(['ps','-eo','pid,ppid,pcpu,pmem,etime,args'],capture_output=True,text=True,check=True).stdout.splitlines()
print('LAUNCH_PROCESSES',json.dumps([line for line in processes if 'scripts.launch_a2_stage1' in line or str(root) in line]),flush=True)
print('HEALTH_COMPLETE',flush=True)
PY
