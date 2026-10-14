#1 +15+a
#th2-q359-proxy12-eval-health-20261014-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/eval_fa4/bin/python3.11 -u - <<'PY'
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-proxy12-eval-20261014-a01')
print('AT',datetime.now(timezone.utc).isoformat())
def tail(p,n=8):
 if not p.exists():return
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-16000));s=f.read().decode(errors='replace')
 print('TAIL',str(p),'\n'+'\n'.join(s.splitlines()[-n:]))
for name in ('preflight.json','supervised/supervisor.json','supervised/run/run.json','supervised/gpus-free-before-training.json','supervised/run/numerics/summary.json','supervised/run/smoke/summary.json','supervised/run/full/summary.json'):
 p=root/name
 if p.exists():
  x=json.loads(p.read_text())
  if name.endswith('smoke/summary.json') or name.endswith('full/summary.json'):
   print('SUMMARY',name,x['status'],x.get('diagnostic'),{x['labels'][k]:v['summary'] for k,v in x['checkpoints'].items()})
  else:print('ARTIFACT',name,json.dumps(x))
for p in sorted(root.glob('audit-*.json')):
 x=json.loads(p.read_text());print('AUDIT',p.name,x['status'],max(t['max_input_tokens'] for t in x['tasks'].values()))
tail(root/'supervisor.log')
for stage in ('numerics','smoke','full'):
 tail(root/'supervised/run'/f'{stage}.log')
 for p in sorted((root/'supervised/run'/stage).glob('**/*.log')):
  if p.stat().st_mtime>datetime.now(timezone.utc).timestamp()-180:tail(p,3)
subprocess.run(['tmux','list-panes','-t',root.name,'-F','#{pane_dead} #{pane_pid} #{pane_current_command}'])
subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv'])
PY
