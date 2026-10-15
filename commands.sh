#1 +15+a
#th2-q359-p7-completed-eval-health-20261015-a02
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'REMOTE'
from pathlib import Path
from datetime import datetime,timezone
import json
from scripts.gpu_status import snapshot
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-p7-completed-eval-20261015-a01')
source=root.parent/'q359-p7-exploration8-20261015-a03'
print('AT',datetime.now(timezone.utc).isoformat())
for p in [source/'user-stop-after-keyonly.json',source/'supervised/supervisor.json',root/'preflight.json',root/'scope.json',root/'supervised/gpus-free-before-training.json',root/'supervised/supervisor.json',root/'supervised/run/run.json',root/'supervised/run/numerics/summary.json',root/'supervised/run/smoke/summary.json',root/'supervised/run/full/summary.json']:
 print('FILE',str(p),p.read_text() if p.exists() else 'MISSING')
for p in [Path(str(root)+'.launch.log'),root/'supervised/run/numerics.log',root/'supervised/run/smoke.log',root/'supervised/run/full.log']:
 if p.exists():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-7000));print('LOG',str(p),f.read().decode(errors='replace'))
print('GPU_SNAPSHOT',json.dumps(snapshot(list(range(8)))))
REMOTE
