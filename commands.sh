#1 +15+a
#th2-q359-p7-exploration-training-health-20261015-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'REMOTE'
from pathlib import Path
from datetime import datetime,timezone
import json
from scripts.gpu_status import snapshot
from scripts.verified_gpu_reclaim import process
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-p7-exploration8-20261015-a03')
run=root/'supervised/run'
print('AT',datetime.now(timezone.utc).isoformat())
state=json.loads((run/'run.json').read_text())
print('QUEUE',state['status'],[(j['name'],j['status']) for j in state['jobs']])
print('RECIPE_GATE',(run/'smoke-recipes.json').read_text())
num=json.loads((run/'numerics/summary.json').read_text())
print('NUMERICS',num['status'],[(c['arm'],c['status']) for c in num['cases']])
for c in num['cases']:
 if c['arm']=='P7-simple-lm-valueonly':print('VALUE_MERGE',json.dumps(c['value_merge_checks']))
folder=run/'a-long/seed-42/A'
config=folder/'train_config.json'
if config.exists():
 c=json.loads(config.read_text())
 print('TRAINING_CONFIG',json.dumps({k:c['training'][k] for k in ('seed','data_seed','max_steps','warmup_steps','per_device_train_batch_size','gradient_accumulation_steps','save_steps','save_total_limit')}))
 print('MODEL_PILOT',json.dumps(c['pilot']))
print('TRAIN_LOG_TAIL',(run/'a-long-train.log').read_text()[-18000:])
print('SUPERVISOR',(root/'supervised/supervisor.json').read_text())
status=snapshot(list(range(8)));print('GPU_SNAPSHOT',json.dumps(status))
workers={p for g in status for p in g['pids']}
print('WORKER_IDENTITIES',json.dumps([process(p) for p in sorted(workers)]))
REMOTE
