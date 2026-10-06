#1 +60+a
#th2-tjx3-p4-four-head-2500-monitor-20261006-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
date -u
/mnt/local/conda-py311/envs/train_env/bin/python3.11 -u - <<'PY'
import json
from pathlib import Path
from scripts.gpu_status import snapshot
root=Path('/mnt/local/_outputs/deep-llms_th2/p4-four-head-2500-20261006-a01')
for name in ['launch.log','preflight.json','supervised/reclaim.json','supervised/gpus-free-before-training.json','supervised/supervisor.json','supervised/run/run.json','supervised/burn-verified.json']:
 p=root/name
 if p.is_file():print('FILE',name,p.read_text()[-12000:],flush=True)
for p in sorted((root/'supervised/run').glob('*.log')):
 print('JOB_LOG',p.name,p.read_text()[-22000:],flush=True)
for folder in sorted((root/'supervised/run/training/seed-42').glob('*')):
 for p in sorted(folder.glob('sdpa-*.json')):
  r=json.loads(p.read_text());print('BACKEND',folder.name,r['phase'],r['runtime']['deterministic_algorithms'],r['has_math'],sorted({op for c in r['calls'] for op in c['forward_operators']}),flush=True)
 p=folder/'result.json'
 if p.is_file():print('RESULT',folder.name,p.read_text(),flush=True)
print('GPUS',json.dumps(snapshot(list(range(8)))),flush=True)
PY
