#1 +15+a
#th2-q359-screen22-results-20261013-a05
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'PYREMOTE'
import json,math
from datetime import datetime,timezone
from pathlib import Path
from scripts.gpu_status import snapshot
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-proxy-12layer-screen22-20261013-a01');run=root/'supervised/run'
print('AT',datetime.now(timezone.utc).isoformat(),flush=True)
queue=json.loads((run/'run.json').read_text())
print('QUEUE',json.dumps(queue),flush=True)
for p in sorted((run/'seed-42').glob('*/result.json')):
 x=json.loads(p.read_text());arm=x['arm'];v=run/f'seed-42-validate-{arm}.json'
 c=json.loads((p.parent/'train_config.json').read_text());state=json.loads((p.parent/'trainer_state.json').read_text())
 logs=[r for r in state['log_history'] if 'loss' in r]
 print('RESULT',json.dumps(dict(arm=arm,step=x['global_step'],status=x['status'],
  validated=v.is_file() and json.loads(v.read_text())['status']=='passed',evaluation=x['evaluation'],
  training_cost=x['training_cost'],last_training=logs[-1] if logs else None,
  train_fingerprint=c['train_fingerprint'],eval_fingerprint=c['eval_fingerprint'],
  seed=c['training']['seed'],data_seed=c['training']['data_seed'])),flush=True)
for j in queue['jobs']:
 if j['status']!='running':continue
 p=run/(j['name']+'.log')
 if p.exists():
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-40000));s=f.read().decode(errors='replace')
  print('CURRENT_LOG',j['name'],'\n'+'\n'.join(s.splitlines()[-18:]),flush=True)
print('SUPERVISOR',(root/'supervised/supervisor.json').read_text(),flush=True)
print('GPUS',json.dumps(snapshot(list(range(8)))),flush=True)
PYREMOTE
