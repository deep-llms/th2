#1 +15+a
#th2-q359-p7-confirmation-health-20261014-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'PY'
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-p7-confirmation3-20261014-a01')
print('AT',datetime.now(timezone.utc).isoformat())
for leaf in ('preflight.json','evaluation-preflight.json','supervised/supervisor.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/run/run.json','supervised/run/numerics/summary.json','supervised/run/summary.json'):
 p=root/leaf
 if p.exists():print('STATE',leaf,p.read_text())
logs=[root/'supervisor.log']
run=root/'supervised/run/run.json'
if run.exists():
 d=json.loads(run.read_text())
 logs.extend(root/'supervised/run'/(j['name']+'.log') for j in d['jobs'][-4:])
for p in logs:
 if p.exists():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-20000));s=f.read().decode(errors='replace')
  print('TAIL',str(p),'\n'+'\n'.join(line[:1600] for line in s.splitlines()[-15:]))
subprocess.run(['tmux','list-panes','-t','q359-p7-confirmation3-20261014-a01','-F','#{pane_dead} #{pane_pid} #{pane_current_command}'])
subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv'])
print('GUARD_DISABLED',Path('/mnt/local/_gpu_guard/DISABLED').exists())
PY
