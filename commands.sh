#1 +15+a
#th2-q359-p7-priority-health-20261015-a01
set -euo pipefail
cd /mnt/local/deep-llms_th2
test "$(hostname)" = thiennh-p6-q359-worker-0
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'PYHEALTH'
import json
from pathlib import Path
from datetime import datetime,timezone
from scripts.gpu_status import snapshot
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-p7-priority15-20261015-a01')
old=Path('/mnt/local/_outputs/deep-llms_th2/q359-p7-tuning22-20261015-a03')
print('AT',datetime.now(timezone.utc).isoformat())
for p in (root/'supervised/run/run.json', root/'supervised/supervisor.json',old/'supervised/run/tg-heads2/seed-42/P7-simple-targetgrad-heads2/result.json'):
 print('STATE',str(p),p.read_text())
p=root/'supervised/run/tg-valueonly-train.log'
print('TRAIN',p.read_bytes()[-6000:].decode(errors='replace'))
print('GPUS',json.dumps(snapshot(list(range(8)))))
PYHEALTH
