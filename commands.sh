#1 +180+a
#th2-tjx3-p4-routing-fix-checks-monitor-20261006-a04
set -euo pipefail
cd /mnt/local/@PROJECT@
date -u
/mnt/local/conda-py311/envs/train_env/bin/python3.11 -u - <<'PYREMOTE'
import json,time
from pathlib import Path
from scripts.gpu_status import snapshot
root=Path('/mnt/local/_outputs/deep-llms_th2/p4-routing-fix-checks-20261006-a02')
for attempt in range(18):
    supervisor=root/'supervised/supervisor.json'
    if supervisor.is_file() and json.loads(supervisor.read_text()).get('finished_at'):break
    time.sleep(10)
for relative in ['launch.log','supervised/reclaim.json','supervised/gpus-free-before-training.json',
                 'supervised/supervisor.json','supervised/run/run.json','supervised/run/numerics.json',
                 'supervised/run/smoke/validation.json','supervised/run/smoke/throughput.json',
                 'supervised/run/complete.json','supervised/burn-verified.json']:
    path=root/relative
    if path.is_file():print('FILE',relative,path.read_text()[-14000:],flush=True)
for path in sorted((root/'supervised/run').glob('*.log')):
    print('JOB_LOG',path.name,path.read_text()[-9000:],flush=True)
for folder in sorted((root/'supervised/run/smoke/seed-42').glob('*')):
    for name in ['component-profile.json','result.json']:
        p=folder/name
        if p.is_file():print('ARM_ARTIFACT',folder.name,name,p.read_text()[-12000:],flush=True)
    for p in sorted(folder.glob('sdpa-*.json')):
        r=json.loads(p.read_text());print('BACKEND',folder.name,r['phase'],r['runtime']['deterministic_algorithms'],r['has_math'],sorted({op for c in r['calls'] for op in c['forward_operators']}),flush=True)
print('GPUS',json.dumps(snapshot(list(range(8)))),flush=True)
PYREMOTE
