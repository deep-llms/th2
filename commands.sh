#1 +60+a
#th2-tjx3-p4-four-head-checks-monitor-20261006-a01
set -euo pipefail
cd /mnt/local/@PROJECT@
date -u
/mnt/local/conda-py311/envs/train_env/bin/python3.11 -u - <<'PYREMOTE'
import json
from pathlib import Path
from scripts.gpu_status import snapshot
root=Path('/mnt/local/_outputs/deep-llms_th2/p4-four-head-checks-20261006-a01')
for relative in ['launch.log','supervised/reclaim.json','supervised/gpus-free-before-training.json',
                 'supervised/supervisor.json','supervised/run/run.json','supervised/run/numerics.json',
                 'supervised/run/smoke/validation.json','supervised/run/smoke/throughput.json',
                 'supervised/run/complete.json','supervised/burn-verified.json']:
    path=root/relative
    if path.is_file():print('FILE',relative,path.read_text()[-14000:],flush=True)
for path in sorted((root/'supervised/run').glob('*.log')):
    print('JOB_LOG',path.name,path.read_text()[-5500:],flush=True)
print('GPUS',json.dumps(snapshot(list(range(8)))),flush=True)
PYREMOTE
