#1 +60+a
#th2-tjx3-p4-four-head-diagnose-monitor-20261006-a02
set -euo pipefail
cd /mnt/local/@PROJECT@
date -u
/mnt/local/conda-py311/envs/train_env/bin/python3.11 -u - <<'PYREMOTE'
import json
from pathlib import Path
from scripts.gpu_status import snapshot
root=Path('/mnt/local/_outputs/deep-llms_th2/p4-four-head-diagnose-20261006-a01')
for relative in ['launch.log','supervised/reclaim.json','supervised/gpus-free-before-training.json',
                 'supervised/supervisor.json','supervised/run/run.json','supervised/run/numerics.json',
                 'supervised/run/smoke/validation.json','supervised/run/smoke/throughput.json',
                 'supervised/run/complete.json','supervised/burn-verified.json']:
    path=root/relative
    if path.is_file():
        value=path.read_text()
        if relative.endswith('numerics.json'):
            data=json.loads(value);bad=data.pop('invalid_gradients',{});data['invalid_count']=len(bad);data['first_invalid']=list(bad.items())[:8];value=json.dumps(data)
        print('FILE',relative,value[-14000:],flush=True)
for path in sorted((root/'supervised/run').glob('*.log')):
    print('JOB_LOG',path.name,path.read_text()[:2200]+'\n...TAIL...\n'+path.read_text()[-1200:],flush=True)
print('GPUS',json.dumps(snapshot(list(range(8)))),flush=True)
PYREMOTE
