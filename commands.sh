#1 +60+a
#th2-tjx3-p4-fa4-checks-monitor-20261006-a04
set -euo pipefail
cd /mnt/local/@PROJECT@
date -u
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'PY'
import json
from pathlib import Path
from scripts.gpu_status import snapshot
root=Path('/mnt/local/_outputs/deep-llms_th2/p4-fa4-checks-20261006-a01')
for name in ['launch.log','preflight.json','supervised/run/numerics.json','supervised/run/cross-backend-P4-iso-4h.json','supervised/run/cross-backend-P4-4h.json','supervised/run/smoke/validation.json','supervised/run/smoke/throughput.json','supervised/run/complete.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/supervisor.json','supervised/run/run.json','supervised/burn-verified.json']:
 p=root/name
 if p.is_file():print('FILE',name,p.read_text()[-12000:],flush=True)
for p in sorted((root/'supervised/run').glob('*.log')):
 print('JOB_LOG',p.name,p.read_text()[-22000:],flush=True)
for folder in sorted((root/'supervised/run/smoke/seed-42').glob('*')):
 for p in sorted(folder.glob('fa4-*.json')):
  print('BACKEND',folder.name,p.name,p.read_text(),flush=True)
 p=folder/'result.json'
 if p.is_file():print('RESULT',folder.name,p.read_text(),flush=True)
print('GPUS',json.dumps(snapshot(list(range(8)))),flush=True)
PY
