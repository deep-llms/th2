#1 +60+a
#th2-q359-p7-exploration-health-20261015-a02
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'REMOTE'
from pathlib import Path
from datetime import datetime,timezone
import json,time
from scripts.gpu_status import snapshot
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-p7-exploration8-20261015-a03')
for attempt in range(45):
 p=root/'supervised/run/run.json'
 state=json.loads(p.read_text()) if p.exists() else {}
 jobs=state.get('jobs',[])
 print('PROGRESS',datetime.now(timezone.utc).isoformat(),state.get('status'),[(j['name'],j['status']) for j in jobs],flush=True)
 if state.get('status')=='failed' or any(j['name']=='a-long-train' for j in jobs):break
 time.sleep(20)
print('AT',datetime.now(timezone.utc).isoformat())
for leaf in ('preflight.json','evaluation-preflight.json','inspection.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/supervisor.json','supervised/run/run.json','supervised/run/numerics/summary.json','supervised/run/smoke-recipes.json','supervised/burn-verified.json'):
 p=root/leaf
 print('STATE',leaf,p.read_text() if p.exists() else 'MISSING')
launch=root.with_name(root.name+'.launch.log')
if launch.exists():print('LAUNCH_LOG',launch.read_text()[-18000:])
for pattern in ('supervised/run/*.log','supervised/run/numerics/*.log'):
 for p in root.glob(pattern):print('LOG_TAIL',str(p.relative_to(root)),p.read_text()[-4000:])
print('GPU_SNAPSHOT',json.dumps(snapshot(list(range(8)))))
print('GUARD_DISABLED',Path('/mnt/local/_gpu_guard/DISABLED').exists())
REMOTE
