#0
#th2-q359-proxy12-remaining-health-20261014-a02
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/eval_fa4/bin/python3.11 -u - <<'PY'
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import json,subprocess
print('AT',datetime.now(timezone.utc).isoformat())
base=Path('/mnt/local/_outputs/deep-llms_th2')
for name in ('q359-proxy12-eval-20261014-a01','q359-proxy12-eval-remaining27-20261014-a01'):
 root=base/name
 print('ROOT',name)
 for leaf in ('supervised/supervisor.json','supervised/burn-verified.json','supervised/run/run.json','supervised/reclaim.json','supervised/gpus-free-before-training.json'):
  p=root/leaf
  if p.exists():print('STATE',leaf,p.read_text())
 p=root/'preflight.json'
 if p.exists():
  d=json.loads(p.read_text());print('PREFLIGHT',d['status'],len(d['checkpoints']),d['accelerate_cache'],d['accelerate_sha256'])
 for stage in ('numerics','smoke','full'):
  path=root/'supervised/run'/stage/'summary.json'
  if path.exists():
   d=json.loads(path.read_text());print('SUMMARY',stage,d['status'],d.get('diagnostic'),len(d.get('checkpoints',{})))
  plan=root/'supervised/run'/stage/'plan.json'
  if plan.exists():
   counts=Counter()
   for j in json.loads(plan.read_text()):
    dest=Path(j['output']);meta=dest/'eval_metadata.json'
    counts[json.loads(meta.read_text()).get('status','unknown') if meta.exists() else 'started' if dest.with_suffix('.log').exists() else 'pending']+=1
   print('GROUP_PROGRESS',stage,dict(counts))
 for leaf in ('supervisor.log','supervised/run/numerics.log','supervised/run/smoke.log','supervised/run/full.log'):
  p=root/leaf
  if p.exists():
   with p.open('rb') as f:f.seek(max(0,p.stat().st_size-10000));s=f.read().decode(errors='replace')
   print('TAIL',leaf,'\n'+'\n'.join(line[:700] for line in s.splitlines()[-5:]))
 print('COMPLETE_MARKER',(root/'supervised/run/complete.json').exists())
subprocess.run(['tmux','list-panes','-t','q359-proxy12-eval-remaining27-20261014-a01','-F','#{pane_dead} #{pane_pid} #{pane_current_command}'])
subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv'])
print('GUARD_DISABLED',Path('/mnt/local/_gpu_guard/DISABLED').exists())
PY
