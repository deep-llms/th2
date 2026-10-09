#1 +60+a
#th2-q359-training-smoke50-status-20261009-a01
set -euo pipefail
/mnt/local/conda-py311/envs/attention_bench/bin/python3.11 -u - <<'PYREMOTE'
import json
from pathlib import Path
from scripts.gpu_status import snapshot
root=Path('/mnt/local/_outputs/deep-llms_th2/q359-proxy-training-smoke50-20261009-a01')
for path in (root/'supervised/supervisor.json',root/'supervised/run/run.json',root/'supervised/run/complete.json'):
    if path.is_file():print(path.name,path.read_text(),flush=True)
print('SUPERVISOR_LOG', '\n'.join((root/'supervisor.log').read_text(errors='replace').splitlines()[-12:]),flush=True)
for path in sorted((root/'supervised/run').glob('*.log')):
    print('JOB_LOG',path.name,'\n'.join(path.read_text(errors='replace').splitlines()[-12:]),flush=True)
for arm in ('A','P6-iso'):
    folder=root/'supervised/run/seed-1042'/arm
    result=folder/'result.json'
    if result.is_file():print('RESULT',arm,result.read_text(),flush=True)
    validation=root/'supervised/run'/('validate-'+arm+'.json')
    if validation.is_file():print('VALIDATION',arm,validation.read_text(),flush=True)
print('GPUS',json.dumps(snapshot(list(range(8)))),flush=True)
PYREMOTE
